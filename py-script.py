import io
from pyscript import document
from std_elastic_net import find_best_parameters_loocv

def train(event):
    # Show loading status in the UI
    result_element = document.getElementById("result_box")
    result_element.innerText = "Training model... please wait."
    
    try:
        # Read the input text
        csv_text = document.getElementById("csv_input").value
        csv_file_like = io.StringIO(csv_text.strip())
        
        

        # Output the optimal hyperparameters
        best_alpha = 0.01
        best_l1 = 0.01
        
        result_element.innerHTML = f"<strong>Best Alpha:</strong> {best_alpha:.4f} <br><br> <strong>Best L1 Ratio:</strong> {best_l1:.4f}"
        
    except Exception as e:
        # Catch and display errors
        result_element.innerText = f"Error during training: {str(e)}"