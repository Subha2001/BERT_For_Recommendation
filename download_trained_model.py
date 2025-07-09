import os
import shutil

def download_trained_model(export_root=None, destination_folder='downloaded_model'):
    """
    Copies the best trained model from the export directory to a user-accessible folder.
    Args:
        export_root (str): Path to the export root where models are saved.
        destination_folder (str): Folder to copy the model to.
    """
    if export_root is None:
        export_root = os.path.join(os.getcwd(), 'experiments')
    model_path = os.path.join(export_root, 'models', 'best_acc_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    os.makedirs(destination_folder, exist_ok=True)
    dest_path = os.path.join(destination_folder, 'best_acc_model.pth')
    shutil.copy2(model_path, dest_path)
    print(f"Model copied to {dest_path}")

if __name__ == "__main__":
    # Example usage: download the model after training
    download_trained_model()
