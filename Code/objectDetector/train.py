from requests.packages import target
from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, model, train_data_loader, trainset, optimizer, device, criterion):
        self.model = model
        self.train_data_loader = train_data_loader
        self.trainset = trainset
        self.optimizer = optimizer
        self.device = device
        self.objectCriterion = criterion


    def train(self, num_epochs=10):
        # EPOCHS = 10
        # train_loss = []
        # train_accuracy = []
        # test_loss = []
        # test_accuracy = []

        # for epoch in tqdm(range(EPOCHS)):
        #     correct = 0
        #     iterations = 0
        #     iter_loss = 0
        #     self.model.train()
        #     for i, (images, labels, bbox) in enumerate(self.train_data_loader):
        #         images = images.to(self.device)
        #         labels = labels.to(self.device)
        #         bbox = bbox.to(self.device)
        #
        #         regressor, classifier = self.model(images)
        #
        #         _, predicted = torch.max(classifier, 1)  ## To get the labels of predicted
        #         predicted_bbox = bbox + regressor  ## to get the bbox of the predicted (add the regression offset with the original bbox)
        #
        #         clf_loss = self.objectCriterion(classifier, labels)
        #         reg_loss = self.bboxCriterion(predicted_bbox, bbox)
        #
        #         total_loss = (clf_loss + reg_loss).clone().detach().requires_grad_(True)
        #
        #         self.optimizer.zero_grad()
        #         total_loss.backward()
        #         self.optimizer.step()
        #
        #         iter_loss += total_loss.item()
        #         correct += (predicted == labels).sum().item()
        #         iterations += 1
        #
        #     train_loss.append(iter_loss / iterations)
        #     train_accuracy.append((100 * correct / len(self.trainset)))
        #     print(
        #         f"Epoch [{epoch + 1} / {EPOCHS}], Training Loss: {train_loss[-1]:.3f}, Training Accuracy: {train_accuracy[-1]:.3f}")

        for epoch in range(num_epochs):  # Iterate over epochs
            self.model.train()
            epoch_loss = 0  # Initialize epoch loss
            num_batches = len(self.train_data_loader)  # Total number of batches

            for batch_idx, (images, targets) in enumerate(self.train_data_loader):
                print(f"Got image {batch_idx} where size of images list {len(images)}")
                # Stack images into a single tensor
                images = torch.stack(images)  # Convert list of tensors to a batch tensor

                # Move images and targets to the appropriate device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                # target_labels = torch.cat([t['labels'] for t in targets], dim=0)

                # Zero gradients
                self.optimizer.zero_grad()
                print(f"Optimizer initialized with zero gradients for epoch {epoch}")

                # Forward pass
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                print(f"Losses computed for epoch:  {epoch}")

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                print(f"backward propagation done for epoch {epoch}")

                # Accumulate loss
                epoch_loss += loss

                # Optionally print batch loss
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{num_batches}], Loss: {loss:.4f}")

            # Average loss for the epoch
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")