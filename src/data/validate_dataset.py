from pathlib import Path

def dataset_exists(path):
    path = Path(path)

    if path.exists():
        print("Dataset exists")
        return True
    else:
        print("Dataset doesn't exist")
        return False

dataset_path = r"D:\good_projects\visionguard\mvtec_anomaly_detection"

print(dataset_exists(dataset_path))

################################################################################
def get_categories(path):
    path = Path(path)

    # Your code here
    categories=[]
    for item in path.iterdir():
        if item.is_dir():
            categories.append(item.name)

    return categories


dataset_path =r"D:\good_projects\visionguard\mvtec_anomaly_detection"

categories = get_categories(dataset_path)

print(categories)

##################################################################################
def validate_category(category_path):
    category_path=Path(category_path)
    train=category_path/"train"
    test=category_path/"test"
    ground_truth=category_path/"ground_truth"
    train_good=train/"good"
    if train.exists() and test.exists() and ground_truth.exists() and train_good.exists():
        return True
    else:
        return False


dataset_path=r"D:\good_projects\visionguard\mvtec_anomaly_detection"
categories=get_categories(dataset_path)
print(categories)

#pick one category

category_path=Path(dataset_path)/categories[0]
print(category_path)
print(validate_category(category_path))

#################################################################################
def validate_all_categories(dataset_path):
    categories = get_categories(dataset_path)

    for category in categories:
        category_path = Path(dataset_path) / category
        result = validate_category(category_path)
        print(category, ":", result)

dataset_path = r"D:\good_projects\visionguard\mvtec_anomaly_detection"

validate_all_categories(dataset_path)
#################################################################################
def count_images(path):
    path=Path(path)
    if not path.exists():
        return 0
    count=0
    image_extensions={".png",".jpg",".jpeg"}
    for item in path.iterdir():
        if item.is_file():
            if item.suffix.lower() in image_extensions:
                count=count+1
    return count
        
dataset_path = r"D:\good_projects\visionguard\mvtec_anomaly_detection"
train_good = Path(dataset_path) / "bottle" / "train" / "good"
print(count_images(train_good))
fake_path = Path(dataset_path) / "bottle" / "does_not_exist"
print(count_images(fake_path))
###################################################################################
from PIL import Image

def is_valid_image(path):
    path=Path(path)
    try:
        image=Image.open(path)
        image.verify()
        return True
    except:
        return False

dataset_path = r"D:\good_projects\visionguard\mvtec_anomaly_detection"
for item in train_good.iterdir():
    if item.is_file():
        print(is_valid_image(item))

#####################################################################################
def validate_images(path):
    path=Path(path)
    total=0
    valid=0
    invalid=0
    image_extensions={".png",".jpg",".jpeg"}

    for item in path.iterdir():
        if item.is_file():
            if item.suffix.lower() in image_extensions:
                total=total+1
                try:
                        image=Image.open(item)
                        image.verify()
                        valid=valid+1
                except:
                        invalid=invalid+1
    return total, valid, invalid

total, valid, invalid = validate_images(train_good)
print("Total:", total)
print("Valid:", valid)
print("Invalid:", invalid)
    