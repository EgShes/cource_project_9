import torch


class Config:
    
    device = torch.device("cpu")

    # facedet params
    max_input_size = 400
    face_size = 160
    man_face_prob = 0.9

    # database params
    db_path = 'db'
    db_name = 'database.db'

    # similarity
    similarity_threshold = 0.55


config = Config()
