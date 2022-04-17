# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from sklearn.neighbors import KNeighborsClassifier

def fetch_train_dataset(categories):
    for file in os.listdir(folder_audio):
        frequencies = ["freq_1", "freq_2", "freq_3", "freq_4", "freq_5", "freq_6", "freq_7", "freq_8", "freq_9",
                       "freq_10", "category"]
        rate, data = wav.read(file)
        fft_out = fft(data)
        name = os.path.basename(file)
        name.replace('.wav', '')

        # data is dataframe?

    return

def fetch_data(folder_audio):
    # Defining model and training it
    categories = ["Cleaning vacum machine", "Cleaning dishes", "Listen music", "Watching TV", "Sleep",
                  "General acticvity"]
    folder_data = "./data/"
    folder_audio = "'./data/audio/"

    f = []
    for (dirpath, dirnames, filenames) in walk(folder_audio):
        f.extend(filenames)
        break
    return f

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder_data = "./data"
    folder_audio = "./data/audio"

    f = fetch_data(folder_audio)

    print(f)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
