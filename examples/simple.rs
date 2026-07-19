/*
This is about as simple as you can get with a network, the arch is
    (768 -> HIDDEN_SIZE)x2 -> 1
and the training schedule is pretty sensible.
There's potentially a lot of elo available by adjusting the wdl
and lr schedulers, depending on your dataset.
*/
use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

use std::fs;
use std::path::{Path, PathBuf};

const HIDDEN_SIZE: usize = 128;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn collect_tdf_files_from(base: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    let entries = match fs::read_dir(base) {
        Ok(entries) => entries,
        Err(_) => return files,
    };

    for entry in entries.flatten() {
        let selfplay_path = entry.path();

        if selfplay_path.is_dir() && 
           let Some(selfplay_name) = selfplay_path.file_name().and_then(|n| n.to_str()) && 
           selfplay_name.starts_with("selfplay_") {

            if let Ok(sub_entries) = fs::read_dir(&selfplay_path) {
                for sub_entry in sub_entries.flatten() {
                    let session_path = sub_entry.path();

                    if session_path.is_dir() {
                        if session_path.is_dir() &&
                           let Some(session_name) = session_path.file_name().and_then(|n| n.to_str()) &&
                           session_name.starts_with("session") {

                            if let Ok(file_entries) = fs::read_dir(&session_path) {
                                for file_entry in file_entries.flatten() {
                                    let file_path = file_entry.path();

                                    if file_path.is_file() && 
                                       file_path.extension().and_then(|e| e.to_str()) == Some("tdf") {
                                        files.push(file_path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    files
}

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        // makes `ntm_inputs` available below
        .dual_perspective()
        // standard optimiser used in NNUE
        // the default AdamW params include clipping to range [-1.98, 1.98]
        .optimiser(AdamW)
        // basic piece-square chessboard inputs
        .inputs(Chess768)
        // chosen such that inference may be efficiently implemented in-engine
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        // map output into ranges [0, 1] to fit against our labels which
        // are in the same range
        // `target` == wdl * game_result + (1 - wdl) * sigmoid(search score in centipawns / SCALE)
        // where `wdl` is determined by `wdl_scheduler`
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        // the basic `(768 -> N)x2 -> 1` inference
        .build(|builder, stm_inputs, ntm_inputs| {
            // weights
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    let schedule = TrainingSchedule {
        net_id: "nn128".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 40,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        save_rate: 10,
    };

    let settings = LocalSettings { threads: 6, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let data_files = collect_tdf_files_from(Path::new("/kaggle/input/datasets/taperihn00/leafselfplay212m/selfplay_serialized_shuffled/"));

    let str_vec: Vec<&str> = data_files
        .iter()
        .filter_map(|path| path.to_str()) 
        .collect();

    // loading directly from a `BulletFormat` file
    let data_loader = loader::DirectSequentialDataLoader::new(
        &str_vec
    );

    trainer.run(&schedule, &settings, &data_loader);
}
