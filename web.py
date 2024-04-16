from flask import Flask, send_file
import subprocess
import threading

app1 = Flask(__name__)
app2 = Flask(__name__)

# Define routes for app1
@app1.route('/')
def index_app1():
    return send_file('index.html')

# Define routes for app2
@app2.route('/')
def index_app2():
    return send_file('index2.html')
def display_csv_files():
    folder_path = '/attendance'  # Replace with the actual path to your CSV files
    max_files_to_display = 10

    # Get a list of CSV files in the specified folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Read up to 10 CSV files and store them in a dictionary
    data_dict = {}
    for i, file in enumerate(csv_files[:max_files_to_display]):
        file_path = os.path.join(folder_path, file)
        data_dict[f'File {i + 1}'] = pd.read_csv(file_path).to_html()
    return render_template_string(template, data_dict=data_dict)

def run_app1():
    app1.run(host='0.0.0.0', port=5001)

def run_app2():
    app2.run(host='0.0.0.0', port=5002)
@app1.route('/work_p')
def work():

    output = subprocess.check_output(['python', 'attendence.py'], text=True)

    return send_file('index.html')

if __name__ == "__main__":
    thread1 = threading.Thread(target=run_app1)
    thread2 = threading.Thread(target=run_app2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()