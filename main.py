from dataset import TikTokDataset
from model import PMMKD



if __name__ == '__main__':
    # Configuration
    config = {
        'embed_dim': 64,
        'prompt_dim': 64,
        'n_layers': 3,
        'batch_size': 2048,
        'lr': 0.001,
        'lambda_p': 0.1,
        'lambda_1': 1.0,
        'lambda_2': 1.0,
        'lambda_3': 0.1,
        'lambda_4': 0.01,
        'teacher_epochs': 1,
        'student_epochs': 1,
        'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    }

    print("Loading dataset...")
    dataset = TikTokDataset()

    print("Initializing PMMKD...")
    model = PMMKD(dataset, config)

    # Train teacher
    model.train_teacher(config['teacher_epochs'])

    model.visualize_user_item_joint(model_type='teacher')

    # Train student with KD
    model.train_student(config['student_epochs'])
    model.visualize_user_item_joint(model_type='teacher')


    # Final evaluation
    print("\nFinal Evaluation:")
    model.evaluate(K=20)
    model.evaluate(K=50)
