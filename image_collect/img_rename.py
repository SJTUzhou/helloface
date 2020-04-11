import os 

def rename_image(path):
    for root, dirs, files in os.walk(path):
        for file, i in zip(files, range(1, len(files)+1)):
            root_splits = root.split('/')
            if '' in root_splits:
                root_splits.remove('')
            new_name = os.path.join(root, root_splits[-1]+'-'+str(i).zfill(3)+'.'+file.split('.')[-1])
            os.rename(os.path.join(root, file), new_name)
