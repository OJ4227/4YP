import pandas as pd

list_of_dims = [[3, 3]]
# list_of_dims = [[2, 2]]
# num_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
num_samples = [50000]

for dims in list_of_dims:
    for num in num_samples:
        # Initialise lists to store the different data frames from different sample files (eg. 100_samples1, 100_samples2...)
        cd_results = []

        # Iterate through each different version of sample which has the same size (num)
        for i in range(1, 2):
            r_file_path = f"C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/"

            # Store the paths to the files containing the results for the different classes of algorithms
            cd_path = r_file_path + f"cd_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"

            # Load in each file for different i and store in the list previously created
            cd_results.append(pd.read_csv(cd_path))

        # Initialise a dict to contain data frames which are the concatenation of the data frames for each value of i
        concat_alg_types1 = {}

        # Store the concatenations for each algorithm class
        # Ignore the 0th column as it doesn't contain useful information

        concat_alg_types1['CD'] = pd.concat(cd_results[i].iloc[:, 1:] for i in range(len(cd_results)))

        # For each concatenated dataframe, calculate the averages, grouped by the alg_name,
        # What you group by will change for each class of algorithm

        for key, concat in concat_alg_types1.items():
            if key == 'CD':
                by_row_index = concat.groupby([concat.columns[0], concat.columns[1]], as_index=False)
                cd_averaged_results = by_row_index.mean()

        # Check if it's computing averages correctly
        # total = 0
        # for i in range(10):
        #     total += data_results[i].iloc[1, 6]
        #
        # mean = total/10

        # Save the averaged results of that particular number of samples to csv
        write_path = f'C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/Results/{dims[0]}x{dims[1]}_{num}_samples/'
        cd_averaged_results.to_csv(write_path + 'cd_averaged_results.csv')
        # Still need to choose the best performing within them
        print(f'Completed {dims[0]}x{dims[1]}_{num}_samples')
