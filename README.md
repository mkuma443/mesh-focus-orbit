# Mesh Focus Orbit

Blender 5.2 用のリトポロジー支援アドオンです。

現在のアドオンバージョン: **3.3.0**

主な機能:

- **通常 MFO**: Object / Edit / Sculpt Mode で、現在のビューポートに見えている MESH のうち画面中央レイで最も手前の面を一時 Orbit 中心にする
- **Face Set MFO (FSMFO)**: 設定した Reference Object の対象 Face Set と、対応する Retopo island だけを一時表示する
- **Smart Face Set Fill**: Sculpt Mode でカーソル直下の外周境界だけをプレビューし、E再押下またはEnterで外周内のseed接続面を Face Set 化する
- Smart Face Set Fill の予測は境界 edge の線だけを描画する。確定時に世代固定したcompact graphの外周境界を越えず、seed接続面を一括で復元して書き込む
- Smart Face Set Fill は、オレンジ形状境界の1〜2面リングだけで実エッジの追加/除去候補を同一スコア比較し、方向に依存せず明確な連続改善だけを採用する（シアン距離境界と保護境界は固定）
- 境界補正の追加側は局所bounded Closing、除去側は補集合の局所bounded Openingとして扱い、シアン接触端の最小guardを除いた内部orange区間だけを評価する（全候補へのN-ring形態処理はしない）
- `Ctrl + E` のgeometry-strict境界補正は物理edgeごとのnormal/crease/contrast/valley/radius信号を実距離近傍でgrayscale max/min filterし、局所median+MADのhysteresisで連続chainだけをproposal化する。分岐では接平面上の明確な直進ペアだけを接続し、曖昧な枝は区間分割する。シアン・hard・radius外では伝播せず、上下のseed方向に依存しない
- 初期にcyanがなくorangeだけの候補は、geometry-onlyの初期baselineを壊さず、全ての安全なorange境界面をmulti-sourceとして通常geometry距離の1〜数metric shellだけlookaheadする。cyan前線が生じる場合だけ暫定bootstrapし、全高signal、hard/hidden/non-manifold/protected/別sheet、frontなしは変更しない
- 初期範囲の決定時だけseedと同じFace Set IDの連結面内をgeodesic bonusで優先する。異なるIDは中立で、非連結同IDへのテレポートや初期範囲確定後のFace Set参照は行わない。初期geometry baselineを先に保持し、same-IDの追加分だけをunionする
- 通常の `E` は画面キャプチャやROI・ピクセルモーフォロジーを行わず、seed周辺の小さな物理距離範囲をprogressive-rangeとして段階的に探索する。既存のface graph/Dijkstra frontierを初回だけ準備し、wheel拡張では新たに露出した外周だけを処理、縮小・再訪では既存距離と候補を再利用する。normal Eはgeometry boundaryを穏やかに扱い、`Ctrl + E` はgeometry-strictを維持する
- 通常Eのwheelは物理半径を単調に増減する。各stageは`progressive-range`のradiusをキーにcacheされ、`newly_processed_faces`、`reused_faces`、cache hit、wheel計算時間をmetricsへ残す。全画面capture・screen morphology・Face Set色は通常Eの候補判定に使わない。描画境界はcompact sliceではなく準備済みfull physical graphとのaccepted/rejected interfaceから復元し、完全なorange/cyan boundaryを保持する
- 初期範囲の決定時だけseedと同じFace Set IDの連結面内をgeodesic bonusで優先する。異なるIDは中立で、後段のwheel・境界補正ではFace Set IDを参照しない。normal Eは同じgeometry baselineを保ち、`Ctrl + E` のstrict判定を変更しない
- `Shift + Alt + E` はSculpt Modeの現在View3Dだけで独立した専用analysis shader表示をON/OFFする。ON時はtoon_dark MATCAP、Face Setカラー（opacity 0.45）、mesh wire overlayを表示し、shading/overlayを完全snapshotする。OFF、load、unregisterで型互換プロパティを冪等復元する。通常Eは表示を変更せず、旧Alt+Eのkeymapは登録しない。`Ctrl + E` は表示を変更しない
- 初期priorの安全判定はcompact analysis sliceの欠落隣接をphysical hardと誤認せず、full geometry由来のphysical degree/hidden/non-manifold metadataを用いる
- 旧screen-space shadow/morphology実装は診断用コードとして残るが、3.3.0の通常E本番経路からは呼び出さない。これによりwheel応答では画面全体のcapture・pixel flood・shader切替を発生させない
- **Topology Colors**: 編集中の選択面へ6色の半透明ガイドを割り当てる
- **Curved Face Set Tube Shape**: Sculpt Mode で seed Face Set の edge-connected tube を自動判別し、曲がった中心線を保ったまま均一化または先細り補正する

## インストール

1. Blender の `Edit > Preferences > Add-ons > Install...` を開く
2. `mesh_focus_orbit.py` を選択する
3. `Mesh Focus Orbit` を有効にする
4. Face Set MFO を使う場合は、Add-on Preferences の `Reference Object` に参照ハイポリメッシュを指定する

Reference Object は、Face Set MFO が使うリトポロジー対象のハイポリメッシュです。通常 MFO は設定値を使わず、現在のビューポートに見えている MESH オブジェクト群を Ray Cast して最も手前の面を Orbit 中心にします。Retopo mesh やその他の Scene mesh も、表示中なら通常 MFO の対象になります。

## 基本操作

初期設定の Activation Key は `Right Shift` です。

| 機能 | 操作 |
| --- | --- |
| 通常 MFO ON/OFF | Object / Edit / Sculpt Mode で設定キーを短時間に2回押す |
| Face Set MFO ON/OFF | Object / Edit Mode で `Ctrl` を押しながら設定キーを短時間に2回押す |
| Smart Face Set Fill プレビュー | Sculpt Mode で `E`、開始Eのrelease後に `E` を再押下（または `Enter`）で確定、ホイールで距離変更、`Esc` で取消 |
| Smart Face Set Fill 厳格プレビュー | Sculpt Mode で `Ctrl + E`、開始Eのrelease後に `E` を再押下（または `Enter`）で確定、ホイールで距離変更、`Esc` で取消 |
| Topology Colors | 編集モードで `Ctrl + Alt + 1..6`（`0`で解除） |
| Curved Face Set Tube Shape | Sculpt Mode で `Ctrl + Alt + T`、ホイールで半径/先細り、`Shift + ホイール`で補正強度、`T`/左クリック/Enterで確定、`Esc`で取消 |
| Local Feature Brush | Sculpt Mode の Asset Shelf から `MFO Local Feature Brush` を選択し、左ドラッグで既存の髪束スケールの山/谷だけを強調、ストローク中のホイールで半径、`Shift + ホイール`で強さ、`Esc`でストローク取消 |

通常 MFO と FSMFO は別の KeyMap Item から直接起動します。FSMFO は外側の非 Undo Trigger を経由せず、Undo 対象の Activation Operator が直接起動し、その中から非 Undo Watcher を開始します。

## 通常 MFO

1. 対象メッシュを画面中央に置く（Object / Edit / Sculpt Mode で利用できます）
2. 設定キーをダブルタップする
3. 画面中央のスクリーン座標から、現在のビューポートに見えている MESH オブジェクト群へ Ray Cast する
4. ワールドレイ距離が最小の面を一時的な Orbit 中心にする
5. Navigation Gizmo または MMB で回転する
6. もう一度設定キーをダブルタップして解除する

マウスカーソル位置ではなく、常に 3D Viewport の画面中央を使用します。Activation 時に一度だけ交点を計算し、MFO 中は中心を再計算しません。

解除時は、開始前に保存した `view_location`、`view_distance`、`view_rotation`、`view_perspective` を復元します。メッシュ、選択状態、3D Cursor、Pivot Point、Object Transform、常設の Orbit Around Selection 設定は変更しません。

右上には通常 `MESH FOCUS ORBIT ON` が表示されます。表示位置や表示自体は Preferences で変更できます。

## Face Set MFO

FSMFO は、Reference Object に `.sculpt_face_set` が存在する場合に使用できます。

1. 作業対象を画面中央に置く
2. `Ctrl + 設定キー`をダブルタップする
3. Reference Object の中央 Ray Hit から Face Set ID を取得する
4. 対象 Face Set だけを表示する一時 Proxy を生成する
5. 元の Reference Object の Viewport 描画を一時的に隠す
6. 必要に応じて Retopo_Work の対象 island だけを表示する
7. Proxy を対象に Orbit する
8. `Ctrl + 設定キー`を再度ダブルタップして解除する

Reference Object は Active Object に変更せず、Retopo_Work は Edit Mode のまま維持します。Reference Object は Shrinkwrap Target として元のまま残り、Proxy は Shrinkwrap Target には使用しません。

### Retopo island isolation

Retopo_Work の Face Set 対応付けは行いません。現在の Retopo mesh の edge-connected component を island として扱います。

FSMFO 起動時、各 island の頂点を Face Set Proxy の BVH へ最近傍投影し、次の複合指標で現在の作業 island を自動判定します。

- `near_ratio`
- median distance
- p90 distance

Face 数ではなく、island の大部分が Proxy に沿っているかを重視します。判定が十分明確でない場合は、Retopo isolation を無理に行いません。対象外の既存 Face、Edge、Vertex は一時的に非表示になります。FSMFO 中に作成された新規 geometry は通常どおり編集できます。

解除時には、開始前の hide 状態を復元します。FSMFO 中に作成された新規要素は表示状態を維持します。

### RetopoFlow compatibility

Preferences の `RetopoFlow Focus-Island Snap/Weld Filter` は既定で OFF です。

ON にした場合、FSMFO 中の PolyPen Snap と Translate Auto Merge に限定して、現在の target island に属すると確認できる既存頂点だけを候補にします。RetopoFlow の標準条件は維持し、FSMFO 以外の RetopoFlow 処理には介入しません。FSMFO 終了時には一時 hook を解除します。

### Undo

FSMFO の Activation は独立した Undo Operator として完了し、Watcher は非 Undo の監視処理として動作します。

- FSMFO 中の通常編集を Undo: 編集操作を戻し、FSMFO は維持する
- Activation 自体を Undo: FSMFO を終了し、Proxy、Reference 表示、Retopo isolation を元へ戻す

Blender 本体の Access Violation を避けるため、Undo の自動実行や自動テストは行いません。

## Smart Face Set Fill

Sculpt Mode でカーソルを Face Set 上へ置き、`E` を押すと局所プレビューを開始します。開始EをreleaseしてからEを再押下、または `Enter` で確定し、`Esc` で取消します。開始キーの押しっぱなしやrepeat、ready前のE/Enterでは確定しません。プレビュー中の左/右クリックは確定・取消に使わず、Sculptのstrokeや選択へ渡しません。

処理は次のように動作します。

- カーソル直下の Face を seed にする
- seed と同じ既存 Face Set ID を使用する
- 共有 Edge と、両端点・向きが一致する未結合の継ぎ目を辿る（メッシュは結合しない）
- 平滑化した法線と複数の範囲の曲率から、段差・谷の境界を検出する
- 凸の頂上は通過し、方向別の凹み判定で付け根の谷を検出する。境界は1近傍分広げる
- 滑らかな領域を取得し、境界の帯は近い領域へ割り当てる。帯から別領域へ再拡張しない
- 非表示面・非多様体エッジは横断しない。rayが非表示面に当たった場合はその面を越えて最初の可視面をseedにする。10万面の固定上限は設けず、探索距離は初期半径の16倍を上限にホイールで調整する
- 対象 Face に既存の seed Face Set ID を直接書き込む
- 固定した複数方向の Lambert 照度近似を使い、まず指定距離の局所 patch を取得してから谷の符号付き変化を検出する。実影や AO、画面の照明は使わず、谷barrierでseed側の元face成分を選び、その周囲を元の面で補正する。半径変更ごとに現在patchから再判定するため、有限長の谷端での正当な再加入を妨げない
- 指定距離内の可視面graphで候補に囲まれた未選択成分だけを補完する。距離境界、欠落edge（crop/メッシュ境界、非多様体、hidden、未接続seam）へつながる成分や別sheetは補完しない
- 現在半径の候補距離内にある有効な共有辺crossingはshape/領域境界（orange）として扱い、fine partition barrierの有無で距離境界（cyan）へ戻さない。距離外または未探索のcrossingは距離境界のままにする
- 外周境界は現在のcompact radius domainから作り、内部閉ループを外部到達ラベルで除外する。外へ開く細い溝、crop/mesh境界、非多様体edge、hidden隣接、未接続seam、別sheetは外周側として保持する。ただし通常は停止境界となる谷でも、現在のブラシ範囲内に対岸が入り連続谷成分が両岸に挟まれる場合は、radius外へ進まず範囲内区間だけを候補へ統合する。この谷救済は全体UV展開ではなく、base候補から同じface graph上で小さいN（最大8）のdilation→erosion Closingを行い、元base境界の離れた2区間以上へ接する新規帯だけを最小Nで統合する。candidateが内包する未選択連結成分は、選択領域との共有境界頂点を除いた外側固有頂点数が100以下の場合だけ全体を統合し、101以上、保護境界、距離外は採用しない
- 境界分類では選択側ではなく隣接する非選択側の距離を使い、seed floodと最終描画のoutside面を一致させる
- 予測は外周境界線だけを表示し、面のtriangulationや面GPU batchを作らない。E再押下またはEnterで世代固定snapshotをseedからfloodし、その面集合を一度だけFace Setへ書き込む。履歴は現在と直前2段階を保持する
- 同期計算中と最新結果の実描画前後に届いたwheel入力は捨て、最新結果の描画後に短い排出区間を経た次のwheelだけを1段階として受け付ける。Escとready済み結果のE/Enter確定は維持する

初回準備では Edit Mode へ切り替えず、全 Face の center と全 loop の edge 次数だけを読み取ります。そこからカーソル seed の Euclidean 範囲と解析用 halo を切り出し、範囲内の polygon の edge、法線、非表示状態だけで局所 CSR、距離、谷・境界判定を作ります。全体の曲率や partition を先に作ってから切り出す処理は行いません。edge の全体次数が2の共有だけを通常接続とし、次数1の継ぎ目は両端点の一致と逆向き、法線の互換性を確認した場合だけ橋渡しします。局所切断境界や非多様体 edge は継ぎ目として扱いません。ホイールで範囲を広げた場合は保持済み配列・面 record を使いながら要求された局所 crop と record を再構築し、縮小と同じ半径への再訪では準備済みの結果を再利用します。

`Ctrl + E` は厳格モードです。探索範囲は通常モードと同じで、境界判定だけを厳しくします。結果が気に入らない場合は、通常の `Ctrl + Z` で操作全体を1ステップ戻してください。

Blender同梱のNumPyを使用します。大規模メッシュでは初回の形状解析に時間がかかります。形状が同じ間は解析結果を再利用し、Sculpt・Undo・接続変更・非表示変更後は再計算します。まず面の向きと接平面からの高さが連続する部分を元の領域へ確保し、盛り上がりが平面側へ侵食するのを抑えます。残る境界帯の所属は面に沿った実距離で決め、その帯の中だけで、広く平滑化した谷の強さと境界の実際の長さを使って線を整えます。画面の明るさは参照せず、視点や照明を変えても同じ境界を使います。メッシュの座標は変更しません。広い途切れや形状上区別できない境界は越える可能性があり、滑らかな領域が全くない部分はクリック面だけを対象にします。

大規模メッシュの初回準備では Edit Mode へ切り替えず、全 Face の center と全 loop の edge 次数だけを読み取ります。カーソル seed の Euclidean 範囲と解析用 halo を切り出し、範囲内の polygon の edge、法線、非表示状態だけで局所 CSR と形状判定を作ります。全体の曲率や partition を先に計算してから切り出す処理は行いません。edge の全体次数が2の共有だけを接続し、次数1の継ぎ目は両端点の一致、逆向き、法線の互換性を確認した場合だけ橋渡しします。局所切断境界と非多様体 edge は継ぎ目として扱いません。ホイール拡大時は保持済み配列・面 record を使って要求された局所 crop と record を再構築し、縮小と同じ半径への再訪では局所データを再利用します。

この機能は Sculpt Mode 専用です。`bpy.ops.sculpt.expand()`、Sculpt Mask、画面の深度や表裏で候補を決める処理、Face Set の新規 ID 生成は使用しません。準備中の処理は内部timerで区切られ、ready前のE再押下やEnterは確定しません。

ショートカットは Blender の `Preferences > Keymap` で `Mesh Focus: Local Face Set Grow` を検索して変更できます。

## Local Feature Brush

Sculpt Mode の Asset Shelf で専用の `MFO Local Feature Brush` asset を選択します。選択は通常のBlender Brush選択として保持され、左ボタンを押した時だけstroke用operatorが開始されます。Asset Shelf、Nパネル、別editor、MMB視点操作はstroke外で通常どおり操作できます。左ボタンを離すと一strokeを一つのUndoステップとして確定し、stroke中の `Esc` はそのstrokeだけを元の座標へ戻します。

専用marker付きのSculptブラシassetをEssentialsから独立した `assets/mfo_local_feature_brush.blend` として配布します。Blender Preferences > File Paths > Asset Libraries でこの `assets` ディレクトリを登録し、Asset Shelfの `MFO Local Feature Brush` を選択してください。Essentialsのassetや標準ブラシは変更しません。径（Blender 5.2の`Brush.size`は直径）と筆圧を読み取ります。N パネルの強さ・対象起伏スケール・半径は各stroke開始時に再読込され、選択中の表示値が次のstrokeへ反映されます。native prep中だけbrush型をDrawへ一時変更し、関連設定はfinallyで復元します。ストローク中だけホイールでこのブラシの半径、`Shift + ホイール`で強さを調整し、待機中のホイールは通常のBlender入力として扱います。専用marker以外ではdispatcherのpollがfalseになり、通常ブラシのLMBとShift Smoothへ介入しません。UI/Asset Shelf領域のクリックと修飾キーは消費せず、Local Feature Brush自身のstroke中はShift入力を変形へ使用せず通過させます。

各dabは画面のray hit周辺だけをBVHから取得し、そのcandidate+haloを圧縮して低域化します。低域化後の符号付き曲率残差へgate、falloff、clampを適用するため、既存の広いridgeは上げ、valleyは深くします。平面・傾斜面・fine-onlyの細波、mask頂点、hidden頂点、半径外は変更しません。全meshのBVH/triangle配列は最初のstrokeだけ段階的に準備し、準備中のrelease/Escで中止できます。途中dabでもMesh.updateとredrawを通知し、同期中の自己更新だけcacheを保持して後続外部更新は破棄します。7.6M頂点での実時間は未検証です。

shared mesh、shape key、Multires、Dyntopo、非一様scale、linked meshは変形前に拒否します。最初のnonzero dabで座標setterの直前に一度だけ、公開RNA schemaを検証した `bpy.ops.sculpt.brush_stroke("EXEC_DEFAULT")` を strength 0 のDrawとして実行し、native Sculpt Undo stepを準備します。外側のstroke operatorは `REGISTER,UNDO` でLMB releaseまでnative pending stepを保持し、selector自体は `UNDO` を持ちません。現在の標準ブラシ型は一時的にDrawへ切り替え、`size`、`use_locked_size`、`unprojected_size`、unified設定、automasking、front-faceなどをsnapshotしてfinallyで復元します。Brush.size hard limitを満たさず、world-space object bounds 8 cornerから作る有限bounding sphereを設定できない場合は変形前に中止します。Undo/Redo、ファイルロード、外部mesh更新ではruntime cacheを破棄し、native履歴後に古い座標snapshotを書き戻しません。Blender 5.2.1の隔離GUIで実本体のAsset Shelf選択・custom stroke・marker選択を維持した実SmoothによるShift交互操作・Undo/Redo・Esc・保護領域・設定復元を検証済みです（attempt034、viewport OpenGL画像）。preview埋め込み後のattempt035では非黒の`production_panel_before.png`にmarker名と色付きpreviewを確認しましたが、後続startup全画面captureには黒画像もあり、継続的な全画面UI合格とは扱いません。実髪、tablet pressure、7.6M頂点性能、全深度surface coverageは未検証で、診断で確認された「Undo後の新strokeをEscすると以前のRedo枝を保持できない」制約は残ります。

## Topology Colors

編集モードで面を選択し、上段の `Ctrl + Alt + 1`〜`6` を押すと、選択した表示中の面へ色番号を保存して半透明オーバーレイを表示します。`Ctrl + Alt + 0` は選択面の色を解除します。RetopoFlow 4 の PolyPen 待機中にもこの機能のキー割当が登録されます。選択を解除した後も色は残り、N パネルの `MFO > Topology Colors` から表示のON/OFF、透明度、割当、解除を操作できます。

色番号はマテリアルを作らず、active Edit Mesh の `mfo_topology_color` FACE 整数属性（0=解除、1〜6=色）へ保存します。.blend、Undo/Redoに含まれます。非表示面、未選択面、別オブジェクトの面、既存マテリアルは変更しません。面の境界と選択中の辺・頂点は読み分けられるように表示します。

## Curved Face Set Tube Shape

Sculpt Mode で Smart Face Set Fill で塗ったチューブへカーソルを置き、`Ctrl + Alt + T` を押します。seed Face Set ID の edge-connected 成分だけを解析し、外部 Face Set 境界が2つなら均一モード A、1つで閉じた尖りへ単調に細くなる場合だけ先細りモード B として予測します。中心線は局所断面から推定するため、曲がったチューブを一本の直線へ置き換えません。円形を前提にせず、カーソル下の実断面 profile（楕円、扁平、軽い不規則形状）を参照します。

予測中は実メッシュを変更せず、実際の候補頂点から間引いた edge を表示します。A はカーソルの world hit 付近の断面形状とサイズを基準に、曲がった中心線へ profile を運びます。B も同じ profile の形を保ったまま根元から閉じた tip へ自然に縮小し、根元と tip の位置を固定します。端部、Face Set 外の頂点、境界共有頂点、hidden、完全 mask は固定し、部分 mask は重みで減衰します。ホイール変更は毎回起動時 snapshot から再計算します。

真の mesh open edge、non-manifold edge、分岐、閉じた先端を一意に確認できない断面は理由を表示して無変更で終了します。Smart Face Set Fill の予測が表示中なら、先にその予測を確定または取消してください。`T` release 後の再押下、左クリック、Enter が一度だけ確定し、右クリックは消費します。`Esc`、mode/object/mesh/topology/visibility/transform の変更、Undo、ファイルロード、アドオン解除では無変更で cleanup します。

## Preferences

`Edit > Preferences > Add-ons > Mesh Focus Orbit` にあります。

- `Enable`: アドオンの有効/無効
- `Activation Key`: 通常 MFO と FSMFO のダブルタップキー。左右の Ctrl / Shift / Alt を選択可能
- `Reference Object`: Face Set MFO が Ray Cast する参照ハイポリメッシュ。通常 MFO はこの設定を使わない
- `Focus Loss Behavior`: Blender がフォーカスを失ったときにモードを維持するか解除するか
- `Double-tap Window`: ダブルタップと判定する時間幅
- `Show Mode Indicator`: MFO/FSMFO の状態表示
- `Debug Display`: Orbit 中心のデバッグポイント表示
- `RetopoFlow Focus-Island Snap/Weld Filter`: FSMFO 中の RetopoFlow Snap/Weld 制限。既定 OFF
- `Topology Colors`: 6色オーバーレイの表示と透明度
- `Local Feature Strength`: 既存の山/谷を強調する強さ
- `Target Feature Scale`: 低域化する髪束スケール
- `Local Feature Radius`: 現在のSculptブラシ径に対する半径倍率

## 制限と復旧

- 通常 MFO は Reference Object が未指定でも、表示中の MESH に画面中央レイが当たれば起動します
- 通常 MFO は画面中央レイが表示中の MESH 群に当たらない場合は起動しません
- FSMFO は表示中の Reference Object が未指定または中央レイを外れる場合は起動しません
- FSMFO には Reference Object の `.sculpt_face_set` が必要です
- RetopoFlow と PolyQuilt の連携機能は、それぞれのアドオンがインストールされている場合だけ有効になります
- ファイルロード、アドオン無効化、ウィンドウ終了時には一時 Proxy、isolation、hook をクリーンアップします

---

# Mesh Focus Orbit — English

A Blender 5.2 add-on for manual retopology workflows.

Current add-on version: **3.3.0**

Main features:

- **Normal MFO**: in Object, Edit, or Sculpt Mode, temporarily orbits around the nearest surface hit on the center ray among visible MESH objects in the current viewport
- **Face Set MFO (FSMFO)**: temporarily shows one Face Set from the configured Reference Object and its matching Retopo island
- **Smart Face Set Fill**: previews only the outer boundary under the cursor and applies the seed-connected faces inside it after a second E press or Enter
- Smart Face Set Fill draws boundary lines during prediction and resolves the generation snapshot into one seed flood at confirmation, preserving distance, hidden, crop, and mesh-domain limits
- Smart Face Set Fill compares add and trim proposals in the one/two-face orange-boundary corridor with one shape score, accepting only a clear continuous real-edge improvement independent of seed direction; cyan distance and protected boundaries remain fixed
- Ordinary E traces maximal physical edge sequences before grayscale Closing. Junctions use the most straight deterministic continuation and split ambiguous pairs; each sequence receives a cumulative physical-distance max-then-min Closing so short gaps in one shadow chain are recovered without joining unrelated branches. The immutable capture-generation barrier map is reused at every radius stage and reports sequence, window, junction, gap, endpoint, and domain metrics
- The former normal-E screen capture/morphology pipeline is retained only as dormant diagnostic code; the active 3.3.0 normal-E path performs no capture or automatic shader switch. Boundary lines come from the full prepared physical graph so compact-slice perimeter edges are not lost
- Ordinary `E` uses the fast progressive-range geometry path. It starts from a small seed-local physical range, prepares the reusable face graph once, processes only the newly exposed outer band on wheel expansion, and reuses cached distances/candidates on shrink or revisit. No viewport capture, ROI pixel flood, screen morphology, or full-mesh projection is performed by the active normal-E path
- Normal-E wheel stages are keyed by physical radius and remain incremental: `progressive-range` metrics report initial/current radius, newly processed faces, reused faces, cache hit, and per-stage compute time. Normal E uses the existing permissive geometry boundary resolver; `Ctrl + E` retains the strict resolver unchanged
- Initial Face Set assistance is limited to the seed-range prior: same-ID faces may receive a geodesic bonus, while different IDs remain neutral. After the initial range is fixed, wheel expansion and boundary refinement do not inspect Face Set IDs or display colors
- `Shift + Alt + E` toggles the independent analysis view for the current Sculpt View3D without starting a fill. It shows the toon_dark MATCAP together with semi-transparent Face Set colors (opacity 0.45) and mesh wire overlay. It snapshots and exactly restores shading/overlay state on toggle-off, load, unregister, or modal error; normal E never changes this manual ownership and Ctrl+E leaves the visible view untouched
- Boundary correction treats add as a bounded local Closing and trim as its bounded complement Opening; only the internal orange interval beyond a minimal cyan-contact guard is evaluated, never a full-candidate N-ring morphology
- `Ctrl + E` geometry-strict boundary correction builds a per-physical-edge normal/crease/contrast/valley/radius signal, applies metric-bounded grayscale max/min filters, and uses median+MAD hysteresis to propose only continuous chains; junctions connect only an unambiguous straight pair in the local tangent plane and split ambiguous arms, while cyan, hard barriers, and the radius boundary stop propagation, independent of seed direction
- Initial Face Set assistance is an additive geometry-only prior: the baseline candidate is computed without Face Set costs, same-ID bonus faces may only be unioned, and the accepted initial mesh-face ids are retained as a monotonic floor for later wheel stages.  An all-orange provisional boundary uses all safe sources for a bounded multi-source metric shell; later refinement and shell propagation do not inspect Face Set ids.
- Shadow luminance/morphology metrics remain available for offline diagnostics only. The active normal-E result reports `progressive-range` path metrics instead; hard, hidden, non-manifold, protected, and separate-sheet safety checks remain active in both modes.
- An initial all-orange candidate with no cyan front preserves a geometry-only baseline, then uses every safe orange boundary face as a multi-source ordinary-geometry metric shell; it bootstraps only when a bounded shell creates a cyan front, leaving all-high-signal, hard/hidden/non-manifold/protected, separate-sheet, and frontless cases unchanged
- During initial range determination only, connected faces sharing the seed Face Set ID receive a geodesic bonus; different IDs remain neutral, no teleport to disconnected components is allowed, and Face Set IDs are not consulted after the initial range is fixed
- Initial-prior safety uses full-geometry physical degree/hidden/non-manifold metadata, never a missing-neighbor count from the compact analysis slice
- **Topology Colors**: assigns six translucent topology guide colors to selected Edit Mesh faces
- **Curved Face Set Tube Shape**: classifies one connected Sculpt Face Set tube and previews curved uniform or tapered shaping
- **Local Feature Brush**: selects a marked Sculpt Brush Asset whose LMB dispatcher amplifies existing broad ridges and valleys locally without creating a new stroke crease

## Installation

1. Open `Edit > Preferences > Add-ons > Install...` in Blender
2. Select `mesh_focus_orbit.py`
3. Enable `Mesh Focus Orbit`
4. If you use Face Set MFO, set the retopology high-poly mesh in the add-on's `Reference Object` field

The Reference Object is the high-poly mesh used by Face Set MFO. Normal MFO ignores this setting and ray-casts the visible MESH objects in the current viewport, choosing the nearest surface along the center ray. Retopo meshes and other scene meshes are also considered when visible.

## Controls

The default Activation Key is `Right Shift`.

| Feature | Shortcut |
| --- | --- |
| Normal MFO ON/OFF | In Object, Edit, or Sculpt Mode, double-tap the configured key |
| Face Set MFO ON/OFF | In Object or Edit Mode, hold `Ctrl` and double-tap the configured key |
| Smart Face Set Fill preview | `E` in Sculpt Mode; release and press `E` again (or press `Enter`) to apply, wheel changes distance, `Esc` cancels |
| Strict Smart Face Set Fill preview | `Ctrl + E` in Sculpt Mode; release and press `E` again (or press `Enter`) to apply, wheel changes distance, `Esc` cancels |
| Topology Colors | `Ctrl + Alt + 1..6` in Edit Mode (`0` clears) |
| Curved Face Set Tube Shape | `Ctrl + Alt + T` in Sculpt Mode; wheel changes radius/taper, `Shift + wheel` changes strength, `T`/left click/Enter applies, `Esc` cancels |
| Local Feature Brush | Select `MFO Local Feature Brush` from the Sculpt Asset Shelf; LMB drag applies one stroke, release commits, `Esc` cancels the current stroke; during that stroke only, wheel changes radius and `Shift + wheel` changes strength |

Normal MFO and FSMFO use separate KeyMap Items. FSMFO is started directly by its Undo-enabled Activation Operator; it does not pass through an outer non-Undo trigger. The Activation Operator starts the non-Undo Watcher and then finishes.

## Normal MFO

1. Place the target mesh at the center of the viewport (available in Object, Edit, and Sculpt Mode)
2. Double-tap the configured key
3. Cast one ray from the viewport center to the visible MESH objects in the current viewport
4. Use the nearest hit along the world ray as the temporary orbit center
5. Orbit with the Navigation Gizmo or MMB
6. Double-tap the configured key again to exit

The mouse cursor position is not used for the hit test. The hit is calculated once on activation and is not recomputed while the mode is active.

On exit, the starting `view_location`, `view_distance`, `view_rotation`, and `view_perspective` are restored. Mesh data, selection, the 3D Cursor, Pivot Point, Object Transform, and the persistent Orbit Around Selection setting are not changed.

The viewport normally shows `MESH FOCUS ORBIT ON`. Its visibility can be changed in Preferences.

## Face Set MFO

FSMFO requires a `.sculpt_face_set` attribute on the Reference Object.

1. Place the work area at the center of the viewport
2. Hold `Ctrl` and double-tap the configured key
3. Read the Face Set ID from the center ray hit on the Reference Object
4. Generate a temporary Proxy showing only that Face Set
5. Temporarily hide the original Reference Object's viewport drawing
6. Optionally isolate the matching Retopo island
7. Orbit around the temporary hit point
8. Hold `Ctrl` and double-tap the configured key again to exit

The Reference Object is not made active, and Retopo_Work remains in Edit Mode. The original Reference Object remains the Shrinkwrap Target; the Proxy is never used as the Shrinkwrap Target.

### Retopo island isolation

Retopo faces are not mapped to Reference Face Set IDs. Each edge-connected component of the current Retopo mesh is treated as an island.

At FSMFO activation, each island is evaluated by nearest-point distances from its vertices to a BVH built from the Face Set Proxy. The composite confidence uses:

- `near_ratio`
- median distance
- p90 distance

The decision emphasizes whether most of an island follows the Proxy, rather than the island's face count. If the best candidate is not sufficiently clear, Retopo isolation is skipped. Existing non-target faces, edges, and vertices are temporarily hidden. Geometry created during FSMFO remains available for editing.

The original hide state is restored on exit. New elements created during FSMFO remain visible.

### RetopoFlow compatibility

`RetopoFlow Focus-Island Snap/Weld Filter` is OFF by default.

When enabled, the filter is limited to PolyPen Snap and Translate Auto Merge while FSMFO is active. Only existing vertices confirmed to belong to the current target island are allowed as candidates. RetopoFlow's standard conditions are preserved, and other RetopoFlow paths are left untouched. Temporary hooks are released when FSMFO ends.

### Undo

FSMFO Activation is a separate Undo-enabled operator that finishes immediately. The Watcher is a non-Undo monitor.

- Undoing an edit made inside FSMFO: the edit is undone and FSMFO remains active
- Undoing the FSMFO Activation: FSMFO ends and the Proxy, Reference visibility, and Retopo isolation are restored

Undo is not executed automatically. This avoids repeating Blender Access Violation scenarios during automated tests.

## Smart Face Set Fill

In Sculpt Mode, place the cursor over a Face Set and press `E` to start the local preview. Release the starting `E`, then press `E` again or press `Enter` to apply; `Esc` cancels. Holding the starting key or its repeat events cannot apply a partial result. Left and right clicks are consumed by the modal preview and do not apply or cancel it.

The algorithm:

- Uses the face under the cursor as the seed
- Reuses the seed's existing Face Set ID
- Traverses shared edges and matching unwelded seams without modifying topology
- Detects steps and valleys using smoothed normals and curvature at multiple scales
- Traverses convex crests, detects concave feet with directional curvature, and expands the boundary by one adjacency step
- Finds the smooth core, then assigns boundary faces to the nearest core without merging cores
- Skips hidden faces and non-manifold boundaries; a ray that first reaches a hidden face continues to the first visible face. There is no fixed 100,000-face cutoff; wheel-adjusted search distance is capped at 16 times the initial radius
- Writes the existing seed Face Set ID directly to the accepted faces
- Uses fixed view-independent multi-direction Lambert illumination as an approximation, not real shadows or ambient occlusion. It acquires the requested local distance patch first, detects signed valley changes there, keeps the seed-side original-face component across valley barriers, and corrects its surrounding band against original faces. Each radius is reevaluated from the current patch so a finite valley may reconnect naturally at its end.
- Labels one compact radius domain from its outside openings to remove inner boundary loops. Open U grooves, crop or mesh boundaries, non-manifold edges, hidden neighbors, unmatched seams, and separate sheets remain outside; a valley normally remains a stopping boundary, but a continuous valley component is included only for the in-range interval whose two shores are inside the current brush range, without traversing beyond the radius. This rescue does not use a full UV unwrap: it applies a small, base-seeded face-graph Closing (dilation followed by erosion, at most eight steps) and merges only new components touching two or more separated arcs of the original candidate boundary. An unselected connected component enclosed by the candidate is merged as a whole only when its unique outer vertices, excluding vertices shared with the selected interface, number at most 100; 101 or more, protected boundaries, and faces outside the distance domain are not adopted.
- Treats a valid shared-edge crossing whose outside face is within the current candidate distance as a shape/region boundary (orange); the fine partition barrier does not demote it to the distance boundary (cyan). Crossings outside the distance or beyond the explored patch remain distance boundaries.
- Uses the non-selected adjacent face for boundary distance and draws only copied boundary lines. No prediction triangulation or face GPU batch is created; confirmation floods the generation-fixed compact graph without crossing its stored outer boundary. History keeps the current and two previous stages.
- Wheel input received during synchronous computation and around the first draw of the latest result is discarded. After the draw callback and a short drain interval, the next wheel is accepted as one stage; Esc and E/Enter confirmation of a ready result remain available.

The first preparation stays in Sculpt Mode. It reads only all face centers and the global degree of every loop edge, then crops a cursor-centered Euclidean range with an analysis halo. Polygon edges, normals, and hidden flags are read only inside that range to build the local CSR, surface distances, valley bands, and boundaries. It does not build full-mesh curvature or partition data and crop it afterward. A normal shared edge is connected only when its global degree is two. A degree-one seam is bridged only after coincident endpoints, reverse winding, and compatible normals are confirmed. Crop boundaries and non-manifold edges are never treated as seams. Wheel expansion reuses retained arrays and face records while rebuilding the requested local crop and records; shrink and revisiting a prepared radius reuse the local data.

`Ctrl + E` enables Strict Mode with the same search extent and stricter boundary decisions. Use normal Blender `Ctrl + Z` to undo the complete operation in one step.

Uses Blender's bundled NumPy. Initial local preparation of large meshes can take several seconds; geometry results are reused until coordinates, topology, transforms or visibility change. Tangent-continuous core extensions are reserved first, using both source-anchored normal and plane-height tolerances; these extensions cannot be reassigned by contour smoothing. This keeps the flat base beyond a raised foot on the base side. Remaining boundary ownership uses physical surface distance, followed by up to six local contour-relaxation sweeps restricted to the boundary band. The energy combines physical boundary length with broadly smoothed concave normal changes; it does not sample screen brightness and is independent of view and lighting. Mesh coordinates are unchanged. Wide gaps or geometrically indistinguishable boundaries can still leak. Components with no smooth core fall back to the clicked face.

This feature is Sculpt Mode only. It does not use `bpy.ops.sculpt.expand()`, Sculpt Mask, screen depth/backface state to choose candidates, or newly generated Face Set IDs. Preparation is divided across internal timer ticks; E/Enter before Ready cannot apply a partial result.

The shortcut can be changed in Blender's `Preferences > Keymap` by searching for `Mesh Focus: Local Face Set Grow`.

## Local Feature Brush

In Sculpt Mode, add the repository's `assets` directory as a Blender Asset Library and select the dedicated `MFO Local Feature Brush` asset from the Asset Shelf. Essentials and built-in brushes are not modified. The marked asset remains selected across strokes as a normal Blender Brush selection; the per-stroke operator starts only on LMB press. The Asset Shelf, N-panel, other editors, and MMB view navigation remain available outside a stroke. Releasing LMB commits one stroke as one Undo step. Pressing `Esc` during a stroke restores that stroke's saved coordinates.

The dedicated marked Sculpt Brush asset is never replaced. Its diameter (`Brush.size` is a diameter in Blender 5.2) and tablet pressure are read for the local dab. N-panel strength, target-feature scale, and radius are re-read at each stroke start, so edits made while the asset remains selected affect the next stroke. During the one-time native prep, only the brush type is temporarily changed to Draw and all related settings are restored in `finally`. While a custom stroke is active, wheel adjusts this brush radius and `Shift + wheel` adjusts strength; while idle, wheel remains ordinary Blender input. The dispatcher is poll-gated by the stable asset marker; standard brush LMB and native Shift Smooth remain outside this explicitly selected tool. UI/Asset Shelf clicks and modifier events are not consumed by the dispatcher. During the custom stroke, Shift is passed through without changing the custom geometry.

Each dab queries only a ray-hit neighborhood from a reusable BVH, then compresses the candidate+halo before low-pass and curvature calculations. A post-low-pass signed-curvature gate, radial falloff, and displacement clamp raise existing broad ridges and deepen valleys while leaving flat/slope/fine-only regions, masked/hidden vertices, and vertices outside the radius unchanged. The initial whole-mesh BVH/triangle read is staged and cancellable; update/redraw is published after changed dabs with a synchronous ownership boundary for cache invalidation. 7.6M-vertex interactive timing is unverified.

Shared mesh data, shape keys, Multires, Dyntopo, non-uniform scale, and linked meshes are rejected before any write. On the first non-zero dab, immediately before the first coordinate setter, the add-on validates Blender 5.2's public `OperatorStrokeElement` schema and runs exactly one `bpy.ops.sculpt.brush_stroke("EXEC_DEFAULT")` with a zero-strength temporary Draw configuration. The outer stroke operator uses `REGISTER,UNDO` so the native pending step remains open until LMB release; the selector itself is non-UNDO. It snapshots and restores the active brush's `size`, `use_locked_size`, `unprojected_size`, unified settings, automasking, front-face, and related fields in `finally`; the finite world-space bounds sphere is derived from all eight object-bound corners, and a Brush.size/unprojected-size hard-limit failure cancels before any write. Undo/Redo, file load, and external mesh updates invalidate runtime cache without writing an old coordinate snapshot over native history. Blender 5.2.1 isolated-GUI testing exercised real Asset Shelf selection, custom strokes, native Shift while the marker remained selected, Undo/Redo, Esc, protection regions, and setting restoration (attempt034, viewport OpenGL evidence). A non-black `production_panel_before.png` also showed the embedded marker preview/name in the Asset Shelf (attempt035), but later startup full-screen captures were black and are not a persistent full-screen UI pass. Real hair, tablet pressure, 7.6M-vertex timing, and all-depth surface coverage remain unverified; the diagnostic proof's known limitation remains that after Undo then starting a new custom stroke, Blender's earlier Redo branch may be unavailable.

## Topology Colors

In Edit Mode, select faces and press the top-row `Ctrl + Alt + 1` through `6` to store a color number on the selected visible faces and draw a translucent overlay. `Ctrl + Alt + 0` clears the selected faces. The feature's keymap entries are also registered while RetopoFlow 4 PolyPen is waiting. Colors remain visible after deselection. Use `MFO > Topology Colors` in the N-panel to toggle display, adjust opacity, assign colors, or clear them.

The color number is stored without creating materials, in the active Edit Mesh's `mfo_topology_color` FACE integer attribute (`0` cleared, `1` through `6` colored). It is included in `.blend` files and Blender Undo/Redo. Hidden faces, unselected faces, other objects, and existing materials are left unchanged. Face boundaries and selected edges and vertices remain distinguishable.

## Curved Face Set Tube Shape

In Sculpt Mode, place the cursor over a tube painted by Smart Face Set Fill and press `Ctrl + Alt + T`. Only the edge-connected component containing the hit Face Set ID is analyzed. Two external Face Set boundaries select uniform mode A; one external boundary selects tapered mode B only when a closed pointed tip and a monotone taper are both supported by the local section evidence. The centerline follows local connectivity and frames, so a curved tube is not flattened onto one straight axis. Circular sections are not required: the actual cursor section profile, including elliptical, flattened, and mildly irregular shapes, is used as the reference.

The preview does not change the mesh. It draws a bounded sample of actual candidate edges from the shaped snapshot. Mode A transports the cursor section's shape and size along the curved centerline. Mode B preserves that profile while shrinking it naturally from the root to the fixed closed tip, with root and tip positions fixed. Boundary and outside vertices, hidden vertices, and fully masked vertices stay fixed; partial Sculpt masks attenuate the correction. Every wheel update is recomputed from the immutable activation snapshot.

Open mesh edges, non-manifold edges, branches, and ambiguous sections are rejected with an explanation and no write. Finish or cancel Smart Face Set Fill before starting. Release the starting `T`, then press `T` again, click left, or press Enter to apply once. Right click is consumed. `Esc`, mode/object/mesh/topology/visibility/transform changes, Undo, file load, and add-on unload clean up without applying a prediction.

## Preferences

Open `Edit > Preferences > Add-ons > Mesh Focus Orbit`.

- `Enable`: Enable or disable the add-on
- `Activation Key`: The double-tap key for Normal MFO and FSMFO; left/right Ctrl, Shift, and Alt are available
- `Reference Object`: The high-poly object used by Face Set MFO ray casts; Normal MFO ignores this setting
- `Focus Loss Behavior`: Keep or exit the mode when Blender loses focus
- `Double-tap Window`: Time window used to recognize a double-tap
- `Show Mode Indicator`: Show the MFO/FSMFO status indicator
- `Debug Display`: Show the temporary orbit-center debug point
- `RetopoFlow Focus-Island Snap/Weld Filter`: Restrict RetopoFlow Snap/Weld candidates during FSMFO; OFF by default
- `Topology Colors`: Toggle the six-color overlay and adjust its opacity

## Limitations and recovery

- Normal MFO can start without a configured Reference Object when the center ray hits a visible MESH
- Normal MFO does not start when the viewport-center ray misses all visible MESH objects
- FSMFO does not start without a visible configured Reference Object or when the center ray misses it
- FSMFO requires the Reference Object's `.sculpt_face_set` attribute
- RetopoFlow and PolyQuilt integration is enabled only when the corresponding add-ons are installed
- Temporary Proxies, isolation state, and hooks are cleaned up during file loading, add-on disable, and window teardown
