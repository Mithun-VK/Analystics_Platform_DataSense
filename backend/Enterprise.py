# check_method.py

import inspect
from app.services.cleaning_service import DataCleaningService  # Adjust the import path as needed

def main():
    methods = [name for name, obj in inspect.getmembers(DataCleaningService) if inspect.isfunction(obj)]
    print("Methods defined in DataCleaningService:", methods)

    if 'clean_dataset' in methods:
        print("Method 'clean_dataset' is inside DataCleaningService")
    else:
        print("Method 'clean_dataset' NOT found in DataCleaningService")

if __name__ == "__main__":
    main()
