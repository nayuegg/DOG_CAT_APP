# 必要なモジュールをインポート
import torch
from animal import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルで推論を行う関数を定義


def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # 学習済みモデルから重み(dog_cat.pt)を読み込み
    net.load_state_dict(torch.load(
        "src/dog_cat.pt", map_location=torch.device("cpu")))
    # データの前処理を定義
    img = transform(img)
    # imgに1次元増やす
    img = img.unsqueeze(0)
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y
# 推論したラベルから犬or猫を返す関数


def getName(label):
    if label == 0:
        return "猫"
    elif label == 1:
        return "犬"
    else:
        return "Error"


# Webアプリの制御部分を定義
# まずはFlaskのインスタンス化
app = Flask(__name__)

# アップロードする拡張子を制限
ALLOWED_EXTENTIONS = set(["png", "jpg", "gif", "jpeg"])

# 拡張子が適切であるかを確認する関数を定義


def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENTIONS

# URLにアクセスがあった時の挙動の定義


@app.route("/", methods=["GET", "POST"])
def predicts():
    # POSTメソッドの時の挙動
    if request.method == "POST":
        # ファイル名がなかった場合の処理
        if "filename" not in request.files:
            return redirect(request.url)
        # ファイルがあった場合、データを取り出してチェックする
        file = request.files["filename"]
        # ファイル名のチェック
        if file and allowed_file(file.filename):
            # 画像ファイルに実行される処理を定義
            # 画像読み込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file)
            # バッファに画像を書き込み
            image.save(buf, "png")
            # Webアプリ上で読み込むべき形に変える
            # 画像データのバイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            # HTML側のソースの記述に合わせるために付帯情報を記述
            base64_data = "data:image/png;base64, {}".format(base64_str)
            # 入力した画像に対して推論を行う
            pred = predict(image)
            animalName_ = getName(pred)
            return render_template("result.html", animalName=animalName_, image=base64_data)
        return redirect(request.url)
    # GETメソッドの時の挙動
    elif request.method == "GET":
        return render_template("index.html")


# アプリを実行するためのコード
if __name__ == "__main__":
    app.run(debug=True)
