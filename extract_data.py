import zipfile
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_av_dir = os.path.join(base_dir, 'data_av')

# تحديد اسم ملف الـ ZIP المضغوط بدقة
zip_name = 'AVDD-images-part-001.zip'
zip_path = os.path.join(data_av_dir, zip_name)

print(f"🔍 بيدور على الملف هنا: {zip_path}")

if os.path.exists(zip_path):
    print("⏳ جارٍ فك الضغط الآن، انتظر لحظات...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_av_dir)
    print("✅ تم فك الضغط بنجاح وزي الفل جوه فولدر data_av!")
else:
    print(f"❌ الملف مش موجود في المسار ده! تأكد إن اسم الملف صح وموجود جوه فولدر data_av.")