from CNN1 import CNN
def main():
    from dataset_loader import load_human_dataset
    X_train, y_train = load_human_dataset()
    cnn = CNN(input_shape=X_train.shape, num_classes=2)
    cnn.train(X_train, y_train, num_epochs=1, learning_rate=0.01)

    from joblib import dump, load

    # Save the trained model
    dump(cnn, 'trained_model.joblib1')
    print("Trained model saved successfully.")

if __name__ == '__main__':
    main()