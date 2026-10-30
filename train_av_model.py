import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class AVDDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        print("🔍 جاري فحص مجلدات الصور (AVDD)...")
        # جمع أول 500 صورة كبداية سريعة عشان نتأكد إن التدريب شغال تمام وما يعلقش
        count = 0
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)
                    self.labels.append(0) # Label افتراضي
                    count += 1
                    if count >= 500: # تحديد عينة للسرعة في البداية
                        break
            if count >= 500:
                break

        print(f"📁 تم تحميل عينة بعدد {len(self.image_paths)} صورة للتدريب السريع.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 الجهاز المستخدم: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_av_dir = os.path.join(base_dir, 'data_av')

    dataset = AVDDDataset(data_dir=data_av_dir, transform=transform)
    if len(dataset) == 0:
        print("❌ لم يتم العثور على صور!")
        return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # استخدام ResNet18 وتحديث الطبقة الأخيرة
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("⏳ بدأ تدريب الـ Epochs الآن...")
    model.train()
    
    for epoch in range(2):  # دورتين تدريب للسرعة والتأكد
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"🔄 الدورة {epoch+1} | الباتش {i+1} | Loss: {loss.item():.4f}")

        print(f"✅ انتهت الدورة (Epoch) رقم {epoch+1}")

    model_save_path = os.path.join(base_dir, 'avdd_depression_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 تم حفظ الموديل بنجاح في: {model_save_path}")

if __name__ == '__main__':
    main()