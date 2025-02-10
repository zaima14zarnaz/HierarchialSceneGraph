class Evaluate():
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_loader = test_dataloader

    def evaluate(self):
        self.model.eval()

