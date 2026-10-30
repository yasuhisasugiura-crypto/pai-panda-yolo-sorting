# PAI Panda YOLO Sorting

真上の RGB-D カメラ画像から、合成データで追加学習した YOLOv8n（6 クラス）が物体を認識し、
その結果だけを使って MoveIt Task Constructor が Panda アームの動作を計画します。
飲み物は左トレイ、食べ物は右トレイへ仕分けます。

- デモ動画：https://youtu.be/rIMcelZ0IlY
- 注記：仕分けは RViz（MoveIt の計画シーン）上で実行しています。Gazebo はカメラ画像の生成に使っています。

## 使用ソフト
- ROS 2 Jazzy
- MoveIt 2 / MoveIt Task Constructor
- Gazebo
- Ultralytics YOLOv8n
- Docker イメージ：tiryoh/ros2-desktop-vnc:jazzy

## フォルダ構成
- src/mtc_tutorial ：仕分けアプリ本体（mtc_node など）
- src/pick_ai ：認識部（YOLO で物体を検出し、深度画像から 3D 位置を計算して仕分けアプリへ渡す）
- tools/ ：データ生成（gen_dataset.py）、学習（train_probe.py）、シーン配置（spawn_scene.py）、評価（j4_probe.py / j4_logits.py / live_eval.py）、ログ判定（check_run.py）
- models/sim6_t2.pt ：学習済み YOLO モデル
- logs/ ：評価・実行ログ

## 実行方法（ターミナルを3つ使います）
各ターミナルは、このリポジトリのいちばん上のフォルダで実行します。

1. ターミナル C：MoveIt を起動

```
ros2 launch moveit_task_constructor_demo demo.launch.py
```

2. ターミナル B：Gazebo＋認識＋仕分けを起動

```
ros2 launch mtc_tutorial pick_place.launch.py use_camera:=true perception_mode:=live3d source_mode:=detections yolo_drives_robot:=true allow_experimental_yolo_control:=true enable_mtc:=true enable_perception:=true autonomous:=true reset_demo_scene:=true detect_wait_sec:=300.0 model_path:=$(pwd)/models/sim6_t2.pt
```

3. ターミナル A：物体を配置

```
python3 tools/spawn_scene.py --seed 960
```

## 結果
- 知覚評価（seed 950〜964 の 15 シーン）：75 個すべて検出、位置誤差 平均 約 0.1cm・最大 0.27cm
- 仕分け（scene 960）：placed 5/5、SKIP・リトライ・計画失敗 0 件、約 118 秒
- 仕分け（scene 951）：placed 4/4、SKIP・リトライ・計画失敗 0 件、約 95 秒

## 限界
- 仕分けは RViz（MoveIt の計画シーン）上での実行で、Gazebo 上の物体は動かない
- 物体は単純な形の 6 クラスのみ、立った状態だけ
