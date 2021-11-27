config = {
    "seed"                                 : 0,
    "result_folder"                        : "result",
    "mode"                                 : ["test", "train"],

    "data.name"                            : "brain_age",
    "data.root_path"                       : "/mnt/data06/umanggup/brain_data/ukbb/",
    "data.train_csv_folder"                : "data/brainage_experiments_configs/uniform_datasize_iid_x8clients/",
    "data.test_csv_path"                   : "data/brainage_experiments_configs/test.csv",
    "data.valid_csv_path"                  : "data/brainage_experiments_configs/test.csv",
    "data.frame_dim"                       : 1,

    "model.name"                           : "regression",

    "model.arch.file"                      : "lib/arch/brain_age_3d.py",
    "model.arch.attn_dim"                  : 32,
    "model.arch.attn_num_heads"            : 1,
    "model.arch.attn_drop"                 : False,
    "model.arch.agg_fn"                    : "attention",

    "train.batch_size"                     : 4,
    "train.optimizer"                      : "sgd",
    "train.lr"                             : 1e-3,
    "train.weight_decay"                   : 0.0,
    "train.log_every"                      : 100,
    "train.learner_evaluation_strategy"    : "per_round",
    "train.community_evaluation_strategy"  : "per_round",
    "train.learner_model_saving_strategy"  : "never",
    "train.community_model_saving_strategy": "per_round",

    "train.num_rounds"                     : 40,
    "train.epochs_per_round"               : 4,
    "test.batch_size"                      : 8,
    }
