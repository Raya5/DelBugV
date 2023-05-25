import pandas as pd
import csv
import os

pd.set_option('display.max_columns', None)
headers = ["CATEGORY", "NETWORK", "PROP", "PREPARE_TIME", "RESULT", "RUN_TIME"]
new_headers = [
  'CATEGORY', 'NETWORK', 'PROP', 'VERIFIER_1', 'PREPARE_TIME_V1', 'RESULT_V1',
  'RUN_TIME_V1', 'VERIFIER_2', 'PREPARE_TIME_V2', 'RESULT_V2', 'RUN_TIME_V2'
]


def get_results():
  # Define the directory path to the root of the repository
  repo_path = "vnncomp2022_results"

  # Loop through each subdirectory in the repository
  results_paths = []
  for dir_name in os.listdir(repo_path):
    # Check if the subdirectory is a directory (not a file) and contains results.csv
    if os.path.isdir(os.path.join(repo_path,
                                  dir_name)) and "results.csv" in os.listdir(
                                    os.path.join(repo_path, dir_name)):
      # If the file exists, print its full path
      results_path = os.path.join(repo_path, dir_name, "results.csv")
      results_paths.append((results_path, dir_name))

  return results_paths


def join_csv_files(file1_path, verifier1, file2_path, verifier2, output_path):
  """
    Joins two CSV files on the first three fields and saves the result to a new CSV file.
    """
  # Load the first file into a pandas DataFrame
  df1 = pd.read_csv(file1_path, header=None, names=headers)

  # Load the second file into a pandas DataFrame
  df2 = pd.read_csv(file2_path, header=None, names=headers)

  # Join the two DataFrames on the first three fields
  joined_df = pd.merge(df1, df2, on=headers[:3], suffixes=('_V1', '_V2'))
  joined_df.insert(3, "VERIFIER_1", verifier1)
  joined_df.insert(7, "VERIFIER_2", verifier2)

  # Write the joined DataFrame to a new CSV file
  with open(output_path, 'a', newline='') as outfile:
    writer = csv.writer(outfile)
    # writer.writerow(headers)
    writer.writerows(joined_df.values)


def check_duplicates(output_path):
  fields = ['CATEGORY', 'NETWORK', 'PROP', 'VERIFIER_1', 'VERIFIER_2']
  df = pd.read_csv(output_path, header=None, names=new_headers)
  duplicates = df.duplicated(subset=fields)

  if duplicates.any():
    print("Duplicates found based on fields:", ", ".join(fields))
    print(df[duplicates])
  else:
    print("No duplicates found based on fields:", ", ".join(fields))


def drop_invalid_rows(input_path, disagreement_path):
  df = pd.read_csv(input_path, header=None, names=new_headers)
  invalid_values = set()

  # remove rows where the 'value' column is not 1 or 2
  cleaned_df = df[df['RESULT_V1'].isin(['sat', 'unsat'])
                  & df['RESULT_V2'].isin(['sat', 'unsat'])].reset_index(
                    drop=True)
  with open("cleaned.csv", 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cleaned_df[[
      'CATEGORY', 'NETWORK', 'PROP', 'VERIFIER_1', 'RESULT_V1', 'VERIFIER_2',
      'RESULT_V2'
    ]].values)

  disagreement_df = cleaned_df[
    cleaned_df['RESULT_V1'] != cleaned_df['RESULT_V2']].reset_index(drop=True)
  with open(disagreement_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(disagreement_df.values)

  # get the set of values that were removed
  invalid_values.update(set(df['RESULT_V1'].unique()) - {'sat', 'unsat'})
  invalid_values.update(set(df['RESULT_V2'].unique()) - {'sat', 'unsat'})

  print(
    f"{len(df)} pairs of verification instances found, {len(df) - len(cleaned_df)} out of which are invalid due to the results: {invalid_values}. \nWhich leaves us with {len(cleaned_df)} valid pairs. \n{len(disagreement_df)} include disagreements - vaild for delbugve. "
  )
  print("Stats:")
  disagreement_df = disagreement_df[
    disagreement_df['VERIFIER_1'].isin(
      [ 'marabou', 'other'])
    & disagreement_df['VERIFIER_2'].isin(
      ['marabou', 'other'])]
  grouped = disagreement_df.groupby(['VERIFIER_1', 'VERIFIER_2'])
  result = grouped.agg(CATEGORY_COUNT=('CATEGORY', pd.Series.nunique),
                       DISAGREEMENT_COUNT=('VERIFIER_1', 'count'))
  print(result.sort_values(by=['DISAGREEMENT_COUNT'], ascending=False))


if __name__ == "__main__":
  output_path = "output.csv"
  disagreement_path = "disagreement.csv"
  open(output_path, 'w', newline='')
  results_pathes = get_results()
  for i in range(len(results_pathes)):
    path1, verifier1 = results_pathes[i]
    for j in range(i + 1, len(results_pathes)):
      path2, verifier2 = results_pathes[j]
      join_csv_files(path1, verifier1, path2, verifier2, output_path)

  check_duplicates(output_path)
  drop_invalid_rows(output_path, disagreement_path)
