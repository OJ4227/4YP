import pandas as pd

list_of_dims = [[2, 2], [3, 3], [2, 4], [4, 2], [4, 4]]
# list_of_dims = [[2, 2]]
num_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
# num_samples = [100]

for dims in list_of_dims:
    for num in num_samples:
        # Initialise lists to store the different data frames from different sample files (eg. 100_samples1, 100_samples2...)
        constraint_results = []
        score_results = []
        hybrid_results = []
        rsmax2_results = []

        # Iterate through each different version of sample which has the same size (num)
        for i in range(1, 11):
            r_file_path = f"C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/"

            # Store the paths to the files containing the results for the different classes of algorithms
            constraint_path = r_file_path + f"constraint_based_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            score_path = r_file_path + f"score_based_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            hybrid_path = r_file_path + f"hybrid_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            rsmax2_path = r_file_path + f"rsmax2_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"

            # Load in each file for different i and store in the list previously created

            constraint_results.append(pd.read_csv(constraint_path))
            score_results.append(pd.read_csv(score_path))
            hybrid_results.append(pd.read_csv(hybrid_path))
            rsmax2_results.append(pd.read_csv(rsmax2_path))

        # Initialise a dict to contain data frames which are the concatenation of the data frames for each value of i
        concat_alg_types1 = {}

        # Store the concatenations for each algorithm class
        # Ignore the 0th column as it doesn't contain useful information
        concat_alg_types1['Constraint'] = pd.concat(constraint_results[i].iloc[:, 1:] for i in range(10))

        concat_alg_types1['Score'] = pd.concat(score_results[i].iloc[:, 1:] for i in range(10))

        concat_alg_types1['Hybrid'] = pd.concat(hybrid_results[i].iloc[:, 1:] for i in range(10))

        concat_alg_types1['Rsmax2'] = pd.concat(rsmax2_results[i].iloc[:, 1:] for i in range(10))

        # For each concatenated dataframe, calculate the averages, grouped by the alg_name,
        # What you group by will change for each class of algorithm
        for key, concat in concat_alg_types1.items():
            if key == 'Constraint':
                by_row_index = concat.groupby([concat.columns[0], concat.columns[1], concat.columns[2]], as_index=False)
                constraint_averaged_results = by_row_index.mean()
            elif key == 'Score':
                by_row_index = concat.groupby([concat.columns[0], concat.columns[1]], as_index=False)
                score_averaged_results = by_row_index.mean()
            elif key == 'Hybrid':
                by_row_index = concat.groupby([concat.columns[0], concat.columns[1], concat.columns[2], concat.columns[3]], as_index=False)
                hybrid_averaged_results = by_row_index.mean()
            elif key == 'Rsmax2':
                by_row_index = concat.groupby([concat.columns[0], concat.columns[1], concat.columns[2], concat.columns[3], concat.columns[4], concat.columns[5]], as_index=False)
                rsmax2_averaged_results = by_row_index.mean()

        # Check if it's computing averages correctly
        # total = 0
        # for i in range(10):
        #     total += data_results[i].iloc[1, 6]
        #
        # mean = total/10

        # Save the averaged results of that particular number of samples to csv
        write_path = f'C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/Results/{dims[0]}x{dims[1]}_{num}_samples/'
        # constraint_averaged_results.to_csv(write_path + 'constraint_averaged_results.csv')
        # score_averaged_results.to_csv(write_path + 'score_averaged_results.csv')
        # hybrid_averaged_results.to_csv(write_path + 'hybrid_averaged_results.csv')
        # rsmax2_averaged_results.to_csv(write_path + 'rsmax2_averaged_results.csv')
        # Still need to choose the best performing within them
        print(f'Completed {dims[0]}x{dims[1]}_{num}_samples')
