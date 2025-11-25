🧠 Human Orientation Detection

A computer vision project to classify human orientation into:
Front
Left
Right
Full Body
Not Applicable (N/A)
Built using Flask, MediaPipe, OpenCV, NumPy.


🚀 Demo Output
Orientation: Left
Orientation: Right
Orientation: Full Body
Orientation: N/A


📦 Requirements
Create virtual environment:
---> python -m venv venv
---> venv\Scripts\activate   # Windows


Install dependencies:
---> pip install -r requirements.txt


▶️ Run the Project

Start the Flask server:
---> python app.py


Open UI in browser:
---> http://127.0.0.1:5000


🧪 API Usage (POSTMAN)

POST:
----> http://127.0.0.1:5000/predict


Body → form-data:
Key: image
Type: File
Value: yourimage.jpg


Response example:
{
  "orientation": "Front"
}


📁 Project Structure
human-orientation/
├── app.py
├── detector.py
├── orientation.py
├── requirements.txt
├── templates/
│   └── index.html
├── README.md
└── .gitignore


📌 Notes
uploads/ folder auto-ignored (local only)
Side face detection smartly handled
Multi-person → N/A
Non-human images → N/A



❤️ Creator: Pratibha Pandey

