import os
import random
import shutil

def combine(audio_dir):
    classes = os.listdir(audio_dir)

    for class_folder in classes:
        class_path = os.path.join(audio_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        train_folder = os.path.join(class_path, 'train')
        test_folder = os.path.join(class_path, 'test')
        
        if os.path.isdir(train_folder) and os.path.isdir(test_folder):
            train_files = os.listdir(train_folder)
            test_files = os.listdir(test_folder)

            for file in train_files:
                src = os.path.join(train_folder, file)
                dst = os.path.join(class_path, file)
                shutil.move(src, dst)

            for file in test_files:
                src = os.path.join(test_folder, file)
                dst = os.path.join(class_path, file)
                shutil.move(src, dst)

            # Remove the train and test folders
            os.rmdir(train_folder)
            os.rmdir(test_folder)



def split(audio_dir):
    # Iterate through the directories inside the audio directory
    for category in os.listdir(audio_dir):
        # Create a path to the current category directory
        category_dir = os.path.join(audio_dir, category)
        
        # Create paths for the train and test directories inside the current category
        train_dir = os.path.join(category_dir, "train")
        test_dir = os.path.join(category_dir, "test")

        # Create the train and test directories if they don't exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Count the number of audio files in the category directory
        audio_files = [file for file in os.listdir(category_dir) if file.endswith(".wav")]
        num_files = len(audio_files)

        # Calculate the number of files for testing and training
        num_test = int(0.35 * num_files)
        num_train = num_files - num_test

        # Shuffle the list of audio files
        random.shuffle(audio_files)

        # Move the files into the train and test directories
        for i, file_name in enumerate(audio_files):
            if i < num_test:
                # Move the file to the test directory
                src_path = os.path.join(category_dir, file_name)
                dst_path = os.path.join(test_dir, file_name)
                shutil.move(src_path, dst_path)
            else:
                # Move the file to the train directory
                src_path = os.path.join(category_dir, file_name)
                dst_path = os.path.join(train_dir, file_name)
                shutil.move(src_path, dst_path)

        # Print the number of files in each train and test directory
        print(f"Category: {category}")
        print(f"Train files: {num_train}")
        print(f"Test files: {num_test}")
        print("----------------------")



if __name__ == "__main__":
    # Specify the path to the main audio directory
    audio_dir = os.path.join(os.getcwd(), "audio")
    action = ['into', 'outof']
    if action == 'outof':
        combine(audio_dir)
    else:
        split(audio_dir)

