

class DataRanker:

    def process_velocity_differences(file_path, focal_fish = 3):
        '''
        The function writes a "Velocity_Differences.csv" file.
        '''

        # Experiment 1
        data = []

        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                if int(row['Experiment_ID']) == 151:
                    data.append(row)
        
        print(f"Total rows from Experiment 151: {len(data)}")
        data.sort(key=lambda row: (int(row['Experiment_ID']), int(row['Agent_ID']), float(row['Kick_Time'])))

        kick_times = {}
        for row in data:
            kick_time = float(row['Kick_Time'])
            if kick_time not in kick_times:
                kick_times[kick_time] = []
            kick_times[kick_time].append(row)

        with open("Velocity_Differences.csv", mode = 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["Agent_ID", "Kick_Time", "Velocity_Agent", "Velocity_Focal_Agent3", "Velocity_Difference"])

                for kick_time in sorted(kick_times.keys()):
                    group = kick_times[kick_time]
                    focal_velocity = None

                    for row in group:
                        if int(row['Agent_ID']) == focal_fish:
                            focal_velocity = float(row['Velocity_Between_Kicks'])

                    if focal_velocity is None:
                        print(f" No focal fish data at Kick_Time {kick_time} — skipping")
                        continue
                        
                    differences = []

                    for row in group:
                        agent_id = int(row['Agent_ID'])

                        if agent_id == focal_fish:
                            continue
                        
                        else:
                            velocity = float(row['Velocity_Between_Kicks'])
                            diff = abs(focal_velocity - velocity)
                            differences.append((agent_id, kick_time, velocity, focal_velocity, diff))

                    differences.sort(key=lambda x:x[4])
                    
            
                    for row in differences:
                        print(f"Writing to CSV: Agent {row[0]}, Kick_Time {row[1]}, Velocity {row[2]}, Focal Velocity {row[3]}, Diff {row[4]}")
                        writer.writerow(row)


    def process_distance_differences(file_path, focal_fish = 3):
        '''
        This function writes a "Distance_Differences.csv" file.   
        
        '''

        # Experiment 1
        data = []

        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                # if int(row['Experiment_ID']) == 151:
                data.append(row)
        
        # print(f"Total rows from Experiment 1: {len(data)}")

        data.sort(key=lambda row: (int(row['Experiment_ID']), int(row['Agent_ID']), float(row['Kick_Time'])))

        experiments = {}
        for row in data:
            experiment = int(row['Experiment_ID'])
            if experiment not in experiments:
                experiments[experiment] = []
            experiments[experiment].append(row)

        with open("Distance_Differences.csv", mode = 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(['Experiment_ID', "Agent_ID", "Kick_Time", "Distance_To_Focal_Agent3", "vx", "vy"])

                for experiment_id, experiment_data in experiments.items():
                    print(f'Experiment_ID: {experiment_id}')
                    
                    kick_times = {}
                    for row in experiment_data:
                        kick_time = float(row['Kick_Time'])
                        if kick_time not in kick_times:
                            kick_times[kick_time] = []
                        kick_times[kick_time].append(row)

                    for kick_time in sorted(kick_times.keys()):
                        group = kick_times[kick_time]

                        sorted_group = sorted(group, key=lambda x: float(x['Distance_To_Focal']))
                        print("________________________________")
                        for row in sorted_group:
                            agent_id = int(row['Agent_ID'])
                            # if agent_id == focal_fish:
                            #     continue

                            vx = float(row['vx'])
                            vy = float(row['vy'])

                            distance = float(row['Distance_To_Focal'])
                            print(f"Kick Time {kick_time} | Agent {agent_id} | Distance: {distance:.4f} | vx: {vx} | vy: {vy}")
                            writer.writerow([experiment_id, agent_id, kick_time, distance, vx, vy])

    def process_distance_differences(file_path, focal_fish = 3):
        '''
        The function writes a "distance_difference_151.csv" file, which processes focal fish with the ID =3 and the relative distance to each fish. 
        '''
        data = []

        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            print(csv_reader.fieldnames)

            for row in csv_reader:
                if int(row['Experiment_ID']) == 151:
                    data.append(row)

        data.sort(key=lambda row: (int(row['Experiment_ID']), int(row['Agent_ID']), float(row['Kick_Time'])))
            
        kick_times = {}  
        for row in data:
            kick_time = float(row['Kick_Time'])
            if kick_time not in kick_times:
                kick_times[kick_time] = []
            kick_times[kick_time].append(row)

        with open("orientation_difference_151.csv", mode = 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(["Agent_ID", "Kick_Time", "Distance_To_Focal", "vx", "vy"])
            

                for kick_time in sorted(kick_times.keys()):
                    group = kick_times[kick_time]

                    sorted_group = sorted(group, key=lambda x: float(x['Distance_To_Focal']))
                    # print("__________________________________")

                    focal_id_data = None
                    for row in sorted_group:
                        agent_id = int(row['Agent_ID'])
                        if agent_id == focal_fish:
                            focal_id_data = row
                            break 

                    if focal_id_data is not None:
                        sorted_group.remove(focal_id_data)
                        sorted_group.insert(0, focal_id_data)

                    for row in sorted_group:
                        agent_id = int(row['Agent_ID'])
                        vx = float(row['vx'])
                        vy = float(row['vy'])
                        distance_difference = float(row['Distance_To_Focal'])
                        writer.writerow([agent_id, kick_time, distance_difference, vx, vy])


    def process_bearing_differences(file_path, focal_fish = 3):
        '''
        This function writes a "Bearing_Differences.csv" file.
        
        '''

        data = []

        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                if int(row['Experiment_ID']) == 151:
                    data.append(row)

        data.sort(key=lambda row: (int(row['Experiment_ID']), int(row['Agent_ID']), float(row['Kick_Time'])))

        experiments = {}
        for row in data:
            experiment = int(row['Experiment_ID'])
            if experiment not in experiments:
                experiments[experiment] = []
            experiments[experiment].append(row)


        with open("Bearing_Differences.csv", mode = 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(['Experiment_ID', "Agent_ID", "Kick_Time", "Bearing_To_Focal_Agent3", 'vx', 'vy'])

                for experiment_id, experiment_data in experiments.items():
                    print(f'Experiment_ID: {experiment_id}')

                    kick_times = {}
                    for row in experiment_data:
                        kick_time = float(row['Kick_Time'])
                        if kick_time not in kick_times:
                            kick_times[kick_time] = []
                        kick_times[kick_time].append(row)


                    for kick_time in sorted(kick_times.keys()):
                        group = kick_times[kick_time]

                        sorted_group = sorted(group, key=lambda x: float(x['Bearing_To_Focal']))
                        print("________________________________")


                        for row in sorted_group:
                            agent_id = int(row['Agent_ID'])
                            if agent_id == focal_fish:
                                focal_id_data = row
                                break 

                        if focal_id_data is not None:
                            sorted_group.remove(focal_id_data)
                            sorted_group.insert(0, focal_id_data)

                        for row in sorted_group:
                            agent_id = int(row['Agent_ID'])                    
                            vx = float(row['vx'])
                            vy = float(row['vy'])
                            bearing = float(row['Bearing_To_Focal'])
                            print(f"Kick Time {kick_time} | Agent {agent_id} | Bearing: {bearing:.4f} | vx: {vx} | vy: {vy}")
                            writer.writerow([experiment_id, agent_id, kick_time, bearing, vx, vy])

    def process_orientation_differences(file_path, focal_fish = 3):
        '''
        This function writes a "Orientation_Differences.csv" file.
        '''

        # Experiment 1
        data = []

        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                if int(row['Experiment_ID']) == 151:
                    data.append(row)
        
        # print(f"Total rows from Experiment 1: {len(data)}")

        data.sort(key=lambda row: (int(row['Experiment_ID']), int(row['Agent_ID']), float(row['Kick_Time'])))
        
        experiments = {}
        for row in data:
            experiment = int(row['Experiment_ID'])
            if experiment not in experiments:
                experiments[experiment] = []
            experiments[experiment].append(row)

        with open("Orientation_Differences.csv", mode = 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["Experiment_ID", "Agent_ID", "Kick_Time", "Orientation_Difference_To_Focal", "vx", "vy"])
                
                for experiment_id, experiment_data in experiments.items():
                    print(f'Experiment: {experiment_id}')
                    
                    kick_times = {}

                    for row in experiment_data:
                        kick_time = float(row['Kick_Time'])
                        if kick_time not in kick_times:
                            kick_times[kick_time] = []
                        kick_times[kick_time].append(row)

                    for kick_time in sorted(kick_times.keys()):
                        group = kick_times[kick_time]

                        sorted_group = sorted(group, key=lambda x: float(x['Orientation_Difference_To_Focal']))
                        print("__________________________________")

                        for row in sorted_group:
                            agent_id = int(row['Agent_ID'])
                            if agent_id == focal_fish:
                                focal_id_data = row
                                break 

                        if focal_id_data is not None:
                            sorted_group.remove(focal_id_data)
                            sorted_group.insert(0, focal_id_data)

                        for row in sorted_group:
                            agent_id = int(row['Agent_ID'])
                            vx = float(row['vx'])
                            vy = float(row['vy'])
                            orientation_difference = float(row['Orientation_Difference_To_Focal'])
                            print(f"Kick Time {kick_time} | Agent {agent_id} | Orientation Difference: {orientation_difference:.4f}| vx: {vx} | vy: {vy}")
                            writer.writerow([experiment_id, agent_id, kick_time, orientation_difference, vx, vy])

    def velocity(file_path):
        kick_times = {}
        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter = ",")

            for row in csv_reader:
                kick_time = float(row["Kick_Time"])
                agent_id = int(row["Agent_ID"])
                vx = float(row["vx"])
                vy = float(row["vy"])
                
                if kick_time not in kick_times:
                    kick_times[kick_time] = []
                kick_times[kick_time].append((agent_id, vx, vy))

        return kick_times
