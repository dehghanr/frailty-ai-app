# Video-Based Frailty Assessment Tool (vFM)

This repository provides the code for a video-based frailty meter (vFM), a lightweight AI-powered web application that enables remote assessment of physical frailty from a 20-second elbow flexion/extension task recorded via webcam.

The tool is based on our 2024 study published in *Biomedical Engineering Letters*, which validates the vFM against a sensor-based frailty meter (sFM) and demonstrates its accuracy, usability, and robustness across various environments.

## 🧠 About the Model

The vFM utilizes Google’s [MediaPipe](https://mediapipe.dev/) framework to extract elbow motion from video and compute biomechanical features—such as slowness, weakness, rigidity, and exhaustion. These features are then used to calculate a frailty index (FI) ranging from 0 to 1. The platform supports both single-task and dual-task conditions.

This tool is especially suited for:
- Telehealth environments
- Remote patient monitoring
- Frailty screening without sensors or wearables

## 🧪 How It Works

1. Upload a 20-second video of elbow flexion/extension (with or without cognitive task).
2. The app extracts elbow joint kinematics using MediaPipe.
3. A frailty index is calculated and made available for download.

For detailed methodology and validation, see our published study below.

## 📖 Citation

If you use this tool in your research, please cite our paper:

> Dehghan Rouzi, M., Lee, M., Beom, J., Bidadi, S., Ouattas, A., Cay, G., ... & Najafi, B. (2024). Quantitative biomechanical analysis in validating a video-based model to remotely assess physical frailty: a potential solution to telehealth and globalized remote-patient monitoring. *Biomedical Engineering Letters*, 1-11.  
> [https://link.springer.com/article/10.1007/s13534-024-00410-2](https://link.springer.com/article/10.1007/s13534-024-00410-2)

## 🚫 Note
The `uploads/` folder is excluded from version control to protect user privacy. `Results/` folders are included but empty.


## 💻 Deployment

You can run this app locally using:

```bash
conda env create -f environment.yml
conda activate your_env_name
python app.py




For help with deploying this app on Render or Namecheap, feel free to reach out.

📬 Contact
For questions or support, contact Mohammad Dehghan Rouzi
📧 mrouzi@mednet.ucla.edu
