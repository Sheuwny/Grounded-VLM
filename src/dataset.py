from datasets import load_dataset

def get_data():
    # Hum RefCOCO dataset ka 'unc' subset use kar rahe hain
    dataset = load_dataset("ydshieh/refcoco", "unc", split="train")
    
    # Ek sample print karke dekhte hain ki data kaisa dikhta hai
    print("Dataset loaded successfully!")
    print(dataset[0])
    return dataset

if __name__ == "__main__":
    get_data()
