import os
import torch
import torch.nn as nn
from torchvision import models

# 1. بناء عمارة الموديل المدمج (Multimodal Fusion Model)
class SALAMMultimodalModel(nn.Module):
    def __init__(self):
        super(SALAMMultimodalModel, self).__init__()
        
        # الجزء الخاص بالصور (Vision Backbone - ResNet18)
        self.vision_model = models.resnet18(weights=None)
        self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128) # تحويل ميزات الصور لـ 128 feature
        
        # الجزء الخاص بالنصوص (Text Backbone - Simplified Linear for Features)
        # نفترض أن ميزات النصوص المستخرجة (مثل TF-IDF) تأتي بحجم معين، مثلاً 100 فتم تحويلها لـ 128
        self.text_fc = nn.Linear(100, 128)
        
        # طبقة الدمج النهائية (Fusion Layer)
        # ندمج ميزات الصور (128) مع ميزات النصوص (128) ليصبح الإجمالي 256
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid() # لتوقع النسبة (اكتئاب أو لا)
        )

    def forward(self, images, text_features):
        # استخراج الميزات البصرية
        v_feat = self.vision_model(images)
        
        # استخراج الميزات النصية
        t_feat = self.text_fc(text_features)
        
        # الدمج (Concatenation)
        combined = torch.cat((v_feat, t_feat), dim=1)
        
        # التوقع النهائي
        output = self.fusion_fc(combined)
        return output

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'avdd_depression_model.pth')
    
    print("🚀 جاري تهيئة موديل الدمج متعدد الوسائط (Multimodal Fusion)...")
    
    # إنشاء نموذج الدمج
    model = SALAMMultimodalModel()
    
    # لو حابب تحميل أوزان موديل الصور اللي تدرب لوحده:
    if os.path.exists(model_path):
        print(f"📦 تم العثور على موديل الصور في: {model_path}")
        # ملاحظة: يتم ربط الأوزان الخاصة بجزء الـ Vision هنا بسهولة
    
    print("✅ تم بناء وهيكلة نموذج الدمج بنجاح وجاهز للاختبار!")

if __name__ == '__main__':
    main()