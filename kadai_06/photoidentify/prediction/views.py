from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# モデルの読み込み（初回のみ読み込むように修正）
model = None
def get_model():
    global model
    if model is None:
        model = VGG16(weights='imagenet')
        # save_model(model, 'vgg16.h5') # save_model は不要
    return model

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            # 4章で、画像ファイル（img_file）の前処理を追加
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)  # VGG16用の前処理
             # 予測の実行
            model = get_model() # モデルを取得
            predictions = model.predict(img_array)

            # 予測結果のデコード
            decoded_predictions = decode_predictions(predictions, top=5)[0] 

            results = [(label, round(probability * 100, 2)) for _, label, probability in decoded_predictions]


            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': results, 'img_data': img_data})
        else:
            # フォームが無効の場合はエラーメッセージなどを確認
            print("Form is not valid")
            for field, error_list in form.errors.items():
                print(f"Error in {field}: {error_list}")

            return render(request, 'home.html', {'form': form})