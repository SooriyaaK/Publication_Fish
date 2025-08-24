

def run_experiment(population_size, p_crossover, m_rate, k, dimensions):
    runs = 20
    max_fit_evals = 1000

    
    data_dict = load_processed_data("features_151.csv")
    orientation_file = orientation_velocity("orientation_difference_151.csv") 

    bias_configurations = [
    {"name": "bias_wall_zone", "bias_function": bias_wall_zone, "flag": True, "loss": 'cosine', "modes": [
    'repulsion_zone',
    'alignment_zone',
    'alignment_domain',
    'repulsion_alignment_zone',
    'repulsion_alignment_domain',
                            ]},
    {"name": "bias_zero", "bias_function": bias_zero, "flag": False, "loss": 'cosine', "modes": [None]},
    {"name": "bias_random", "bias_function": bias_random, "flag": True, "loss": 'cosine', "modes": [None]},
    {"name": "bias_wall", "bias_function": bias_wall, "flag": True, "loss": 'cosine', "modes": [None]},
    
    {"name": "bias_positive", "bias_function": bias_positive, "flag": True, "loss": 'cosine', "modes": [None]},
    {"name": "bias_negative", "bias_function": bias_negative, "flag": True, "loss": 'cosine', "modes": [None]},
    ]

    for config in bias_configurations:
        loss = config['loss']
        bias_func = config['bias_function']
        bias_flag = config['flag']
        name = config['name']

        for mode in config['modes']:
            print(f"\nRunning: {loss} {name}, flag={bias_flag}, mode={mode}")

            all_runs_f = []
            all_runs_x = []

            for _ in range(runs):
                x_best, f_best = ea(
                    population_size,
                    max_fit_evals,
                    p_crossover,
                    m_rate, 
                    k, 
                    mode,
                    dimensions,
                    bias_flag, 
                    bias_func, 
                    orientation_file,
                    data_dict,
                    loss,
                    objective_function,
                )
                
                all_runs_f.append(f_best)
                all_runs_x.append(x_best)
        
            filename = f"{loss}_results_{name}"
            if mode:
                filename += f"_{mode}"
            filename += ".csv"
            
            save_to_csv(all_runs_x, all_runs_f, filename = filename)

            plot_fitness_over_time(all_runs_f)

    return all_runs_f, all_runs_x

all_runs_f, all_runs_x = run_experiment(population_size=40, p_crossover=0.6, m_rate=0.2, k=8, dimensions = 6)
