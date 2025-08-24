

def objective_function(bias_flag, bias_func, velocity_file, data_dict, mode, dim, loss, weights, focal_agent = 3, experiment_id = 151):  
   
    predicted_positions_list = []
    actual_positions_list = []

    if not bias_flag:
        weights = weights[:5]

    if not bias_flag:
        w1, w2, w3, w4, w5 = weights
    else:
        w1, w2, w3, w4, w5, w6 = weights
            
    alpha, beta, gamma, delta, epsilon = w1, w2, w3, w4, w5

    if bias_flag:
        bias = w6
    else:
        bias = 0.0

    weights = np.array([alpha, beta, gamma, delta, epsilon, bias])
        
    if np.count_nonzero(weights) > 0:
        norm = np.linalg.norm(weights, ord = 1)
        weights = weights / norm

    kicktimes = []
    mse_loss = 0.0
    
    for key in velocity_file:
        kicktimes.append(key)

    for agent_id, vx, vy in velocity_file[kicktimes[0]]:
        if agent_id == focal_agent:
            predicted_velocity = np.array([vx, vy])
            break
        
    # inital position at timestep 0 
    first_kicktime = np.min(kicktimes)
    initial_x = data_dict[first_kicktime]['x_coords'][0]  # focal agent is at position 0
    initial_y = data_dict[first_kicktime]['y_coords'][0]

    predicted_position = np.array([initial_x, initial_y])

    for i in range(len(kicktimes)-1):

        kicktime = kicktimes[i]
        next_kicktime = kicktimes[i + 1]
        
        velocities = velocity_file[kicktime]
        next_velocities = velocity_file[next_kicktime]

        updated_velocity = []
           
        for j, (agent_id, vx, vy) in enumerate(next_velocities):
            if j == 0:
                updated_velocity.append((agent_id, predicted_velocity[0], predicted_velocity[1]))
            else:
                updated_velocity.append((agent_id, vx, vy))
           
        vx_total = 0.0
        vy_total = 0.0   
            
        for j, (agent_id, vx, vy) in enumerate(updated_velocity):
            weight = weights[j]
          
            vx_total += vx * weight  
            vy_total += vy * weight 

        if bias_flag:
            bias = bias_func(data_dict, kicktime, predicted_velocity, mode)
            vx_total += bias[0] * weights[5]  # Multiply by the bias agent's weight
            vy_total += bias[1] * weights[5]  # Multiply by the bias agent's weight
        
        predicted_velocity = np.array([vx_total, vy_total]) 

        # update

        if loss == 'cosine':
            actual_velocity = np.array([
            data_dict[next_kicktime]['vx_list'][0],
            data_dict[next_kicktime]['vy_list'][0]
        ])

            dotproduct = np.dot(actual_velocity, predicted_velocity)

            magnitude_a = np.linalg.norm(actual_velocity)
            magnitude_b = np.linalg.norm(predicted_velocity)

            cosine_similarity = dotproduct / (magnitude_a * magnitude_b)

            mse_loss += 1 - np.round(cosine_similarity, 4)
         
        else:

            if loss == 'mse_kicktime':
                # per kicktime mse loss
                updated_x = data_dict[kicktimes[i]]['x_coords'][0]  # focal agent is at position 0
                updated_y = data_dict[kicktimes[i]]['y_coords'][0]

                predicted_position = np.array([updated_x, updated_y])
                predicted_position += predicted_velocity
            
            elif loss == 'mse_trajectory':
                # trajectory mse loss
                predicted_position += predicted_velocity


            actual_position = np.array([
                data_dict[next_kicktime]['x_coords'][0],
                data_dict[next_kicktime]['y_coords'][0]
            ])

            delta_x = predicted_position[0] - actual_position[0]
            delta_y = predicted_position[1] - actual_position[1]
            distance = np.sqrt(delta_x**2 + delta_y**2)

            mse_loss += distance** 2

    return mse_loss