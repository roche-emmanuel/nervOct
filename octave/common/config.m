function cfg = config()
% This method is used to build the configuration scrip which will be used during the tests.

% Source of all our data:
cfg.datapath='../data/EUR_21_10_2014';

% Name of the symbol to handle used as a pattern to match files:
cfg.symbol_name='EUR'; 

% week price dataset name:
cfg.week_dataset_name='week_datasets';

% Number of symbol pairs used to build the complete dataset:
cfg.num_symbol_pairs = 6;

% limit number of minutes to consider per week:
cfg.min_week_minutes = 60;
cfg.max_week_minutes = 120; % Note that that one is measured as offset from the maximum number of minutes available in a week.

% total number of minutes in a given week of trading (not counting the weekend).
total_mins = 5*1440-1;

% minutes indices used for each week of data:
cfg.minute_indices = (cfg.min_week_minutes:(total_mins - cfg.max_week_minutes))';

% Total number of minutes considered ber week:
cfg.num_week_minutes = size(cfg.minute_indices,1);

% Number of input minute bars to consider for each example when building feature matrix for instance:
cfg.num_input_bars=60;

% week feature dataset name:
cfg.week_feature_pattern=sprintf('features_%d/week_%%d',cfg.num_input_bars);

% Number of bars to consider when building a prediction:
cfg.num_pred_bars=5;

% dataset ratios for training session:
% given in the order: train/cv/test.
% The some should be 1.0;
cfg.dataset_ratios = [0.75 0.25 0.0];

cfg.spread=0.00008;
cfg.min_gain=cfg.spread*1.2;
cfg.max_lost=cfg.spread*0.5;

% Target symbol pair to train on:
% This should be a valid index in the range 1:cfg.num_symbol_pairs
cfg.target_symbol_pair = 6; % 6 = EURUSD symbol.

% Default number of max iterations to perform:
cfg.default_max_training_iterations = 50;

% Default regularization parameter:
cfg.default_regularization_parameter = 0.0;

% Number of features to consider:
cfg.num_features = 1 + 4 * cfg.num_symbol_pairs * cfg.num_input_bars;

% Deep training status:
cfg.default_deep_training = false;

% Apply shuffle on the training data:
cfg.shuffle_training_data = true;

% Default RMS stop value to use when performing CUDA training:
cfg.default_rms_stop = 0.002;

% File used to save the learning curve:
cfg.learning_curves_graph_file = '../data/results/learning_curves.png';

% Define if we should use CUDA training or not:
% cfg.use_CUDA = true;
cfg.use_CUDA = true;

% verbose outputs:
cfg.verbose = true;

% Use PCA:
cfg.use_PCA = true;

% Quantity of variance to retain when performing PCA: (in percent)
cfg.PCA_variance = 99.9;

% Apply early stopping when training the networks:
cfg.use_early_stopping = false;

% Use sparse initialization:
cfg.use_sparse_init = false;

% Number of neurons with active weights when using sparse initialization:
cfg.sparse_init_lot_size = 15;

end
