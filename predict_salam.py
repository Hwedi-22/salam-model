import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1. نفس عمارة الموديل المدمج
class SALAMMultimodalModel(nn.Module):
    def __init__(self):
        super(SALAMMultimodalModel, self).__init__()
        self.vision_model = models.resnet18(weights=None)
        self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)
        self.text_fc = nn.Linear(100, 128)
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, images, text_features):
        v_feat = self.vision_model(images)
        t_feat = self.text_fc(text_features)
        combined = torch.cat((v_feat, t_feat), dim=1)
        output = self.fusion_fc(combined)
        return output

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'avdd_depression_model.pth')

    print("🔍 جاري تحميل الموديل المتكامل للتنبؤ...")
    model = SALAMMultimodalModel().to(device)
    
    # تحميل الأوزان اللي اتدربت عليها
    if os.path.exists(model_path):
        # تحميل أوزان الـ Vision المبدئية
        state_dict = torch.load(model_path, map_location=device)
        model.vision_model.load_state_dict(state_dict, strict=False)
    
    model.eval()

    # تجهيز صورة تجريبية من مجلد الصور عندك
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # البحث عن أي صورة في مجلد data_av للاختبار
    sample_img_path = None
    for root, dirs, files in os.walk(os.path.join(base_dir, 'data_av')):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_img_path = os.path.join(root, file)
                break
        if sample_img_path:
            break

    if not sample_img_path:
        print("❌ لم يتم العثور على صور تجريبية في مجلد data_av.")
        return

    print(image := Image.open(sample_img_path).convert('RGB'))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # إنشاء ميزات نصية وهمية للتجربة (حجمها 100 قيمة زي ما عرفنا الموديل)
    text_tensor = torch.randn(1, 100).to(device)

    # التنبؤ
    with torch.no_grad():
        output = model(image_tensor, text_tensor)
        depression_score = output.item()

    print("\n" + "="*40)
    print(f"📊 نتيجة التنبؤ لمشروع SALAM Multimodal:")
    print(f"🖼️ الصورة المستخدمة: {os.path.basename(sample_img_path)}")
    print(f"📈 نسبة احتمال الاكتئاب (Depression Score): {depression_score:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    predict()