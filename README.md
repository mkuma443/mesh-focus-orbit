# Mesh Focus Orbit

Blender 5.2 用のリトポロジー支援アドオンです。

アドオンバージョン: **3.3.36**

## 主な機能

- **通常 MFO**: Object / Edit / Sculpt Mode で、現在の 3D Viewport の中央レイが最初に当たる表示中の MESH 面を一時的な Orbit 中心にします。
- **Face Set MFO (FSMFO)**: Reference Object の中央レイが当たる Face Set と、対応が明確な Retopo island だけを一時表示します。
- **Guided Ridge**: Sculpt Mode で固定したcrest guideと左右rail幅を指定し、幅内の既存protrusionだけを内側へplaningします。凹みを埋めず、guide・anchor・裏面は固定します。
- **Smart Fill**: Sculpt Mode では Face Set、Vertex Paint ではアクティブな FLOAT_COLOR/BYTE_COLOR の POINT/CORNER 属性を使って、カーソル下の局所領域をプレビューし、確定時に同じシード色を書き込みます。
- **Shadow Analysis View**: `Shift + Alt + E` で現在の View3D だけに独立した解析表示を切り替えます。
- **Topology Colors**: Edit Mode の選択面に 1〜6 の半透明ガイド色を保存します。
- **Curved Face Set Tube Shape**: 曲がった Face Set チューブを局所断面に沿って均一化または先細り補正します。

## インストール

1. `Edit > Preferences > Add-ons > Install...` を開きます。
2. `mesh_focus_orbit.py` を選択します。
3. 配布物の `assets/mfo-toolbar-icons-astra/` を、アドオンファイルと同じ配置先の相対 `assets/mfo-toolbar-icons-astra/` へコピーします（欠落時はBlender標準アイコンへフォールバックします）。
4. `Mesh Focus Orbit` を有効にします。
5. FSMFO を使う場合は、Add-on Preferences の `Reference Object` に Face Set を持つ参照メッシュを指定します。

アドオンはユーザー設定以外のシーンを自動保存しません。Reference Object は FSMFO 専用です。通常 MFO は現在の View3D で表示中の MESH 群を中央レイで調べ、最前面の hit を対象にします。Guided Ridge は現在のアクティブな MESH を対象にします。

Tツールバーは、編集可能なAstra Blenderソースから生成した5種のBlender VCO `.dat` ジオメトリアイコンを使用します。`assets/mfo-toolbar-icons-astra/` をアドオンの相対パスへ保持してください。欠落時は機能ごとにBlender標準アイコンへ安全にフォールバックします。

### TツールバーとNサイドバー

左側のTツールバーにはモード別のMFO常駐ツールが表示されます。`MFO: Focus Surface` を選んでメッシュ面を左クリックすると、クリック座標直下を通常MFOの対象にします。Object/Edit Modeでは同じツールの `Ctrl + 左クリック` が、従来のCtrl起動と同じFace Set MFO（厳密）経路になります。Face Set MFO専用、Smart Fill（Ctrlで厳格）、Guided Ridge、Tube Shapeも対応モードに表示されます。Smart FillはSculpt ModeではFace Setを、Vertex Paintではアクティブ色属性を対象にします。

右側の `MFO` タブでは、Preferencesを開かずに `Reference Object` と日常設定を変更できます。Reference ObjectはPreferencesと同じSceneプロパティを編集するため、設定値は二重化されません。クリック座標は常にイベントの3D View WINDOW region座標を使用し、画面中央や過去のブラシ位置へ置き換えません。

## 基本操作

| 機能 | モード | 開始・確定 | 取消・補足 |
| --- | --- | --- | --- |
| 通常 MFO | Object / Edit / Sculpt | 設定した Activation Key を短時間に 2 回 | 同じ操作で終了 |
| Face Set MFO | Object / Edit | `Ctrl` + Activation Key を短時間に 2 回 | 同じ操作で終了 |
| MFO Tツール | Object / Edit / Sculpt | Tツールで選択後、面を左クリック（Object/EditはCtrlでFace Set） | 同種入口を3D View region内で再クリックしてトグルOFF。Face Set ON中はFace Set入口を使う |
| Smart Fill Tツール | Sculpt / Vertex Paint | ツールで選択後、面を左クリック（Ctrlで厳格） | 開始クリックを離した後、次のLMBで確定。以降は既存Smart Fillモーダルの操作 |
| Guided Ridge / Tube Shape Tツール | Sculpt | ツールで選択後、対象面を左クリック（Guided Ridgeの最初のクリックは起点設定のみ） | Guided Ridgeは準備・計算中に進捗を表示し、`Esc`/右クリックで取消。準備完了後にLMBで点追加、Wheel/Shift+Wheelで左右共通幅（requested max）を調整、`Enter`で適用 |
| Guided Ridge | Sculpt | `Ctrl + G`。カーソル下の面を起点にし、LMB で点を追加、`Enter` で適用 | `Backspace` で直前の点を削除、`Esc` または右クリックで取消 |
| Guided Ridge Repeat Last | Sculpt | Blender 標準の `Shift + R` | 保存済みguide/widthを現在の表面へ再投影して独立Undoで適用。専用Shift+R keymapはありません |
| Smart Fill | Sculpt / Vertex Paint | `E`、開始キーを離してから `E` または `Enter` | SculptはFace Set、Vertex Paintはアクティブ色属性。ホイールで距離、`Esc` で取消 |
| Strict Smart Fill | Sculpt / Vertex Paint | `Ctrl + E`、開始キーを離してから `E` または `Enter` | SculptはFace Set、Vertex Paintはアクティブ色属性。ホイールで距離、`Esc` で取消 |
| Shadow Analysis View | Sculpt | `Shift + Alt + E` | 現在の View3D の表示補助だけを切替 |
| Topology Colors | Edit | `Ctrl + Alt + 1`〜`6` | `Ctrl + Alt + 0` で選択面の色を解除 |
| Curved Face Set Tube Shape | Sculpt | `Ctrl + Alt + T`、`T`、LMB、または `Enter` | `Esc` または右クリックで取消。ホイールで半径・先細り、`Shift + ホイール`で補正強度 |

各モーダル機能は開始時のオブジェクト、メッシュ、モード、可視性、トポロジーを監視します。これらが外部操作で変わった場合は、適用せず安全に終了します。

## 通常 MFO

Activation Key の初期値は `Right Shift` です。通常 MFO は 3D Viewport の中央座標から一度だけレイキャストし、表示中の MESH のうち最も手前の面を一時的な Orbit 中心にします。Navigation Gizmo や MMB で視点を回転できます。

終了時には開始前の `view_location`、`view_distance`、`view_rotation`、`view_perspective` を復元します。メッシュ、選択状態、3D Cursor、Pivot Point、オブジェクト変換、常設の Orbit Around Selection 設定は変更しません。

## Face Set MFO

FSMFO は Reference Object の `.sculpt_face_set` を使います。中央レイの Face Set を一時 Proxy として表示し、Retopo mesh は edge-connected component (island) 単位で、Proxy への距離と一致率が十分明確な場合だけ対象 island を絞ります。判定が曖昧な場合は Retopo isolation を行いません。

Reference Object をアクティブオブジェクトへ切り替えず、Retopo_Work の Edit Mode も維持します。終了時には開始前の表示・isolation 状態を復元し、FSMFO 中に作られた新規 geometry の表示状態は維持します。

Preferences の `RetopoFlow Focus-Island Snap/Weld Filter` を有効にすると、FSMFO 中に記録した target island に属する既存頂点だけへ、PolyPen Snap と Translate Auto Merge の候補を制限します。RetopoFlow がない場合や FSMFO 外では何もしません。

## Guided Ridge

この節は3.3.36のGuided Ridge動作を説明します。

Guided Ridge（3.3.36）は、固定したguide crestと、Wheelで指定する左右共通planing半幅（requested max）から2枚のtarget planeを作り、幅内でtargetより外側へ突出した既存surfaceだけを内側へ削ります。毛先などsurfaceの可用幅が狭い区間では左右を独立に実surfaceへtaperし、別sheetへ飛ばず、表示railと適用railを一致させます。盛り上げや凹みの埋め戻しは行わず、guide、左右rail anchor/support、幅外、裏面、Face Set boundary、hidden、full maskは固定します。partial maskは削り量を減衰します。railとcandidateは同じcaptured width/frame resultを共有し、投影不能区間や明確な別sheet競合は無変更で拒否します。準備とEnter後の計算はtimerで進捗表示し、`Esc`/右クリックで取消できます。各railの実投影q/hとbarycentric supportも共通結果を使うため、投影が公称幅からずれても表示railを通るtargetになります。通常MFO/Face Set MFOがONでも状態を変更せず開始できます。ガイド編集中のMMB/NDOF/NUMPAD/Navigation GizmoはBlenderの視点操作へ透過します。

1. Sculpt Mode で対象 Face Set 上にカーソルを置き、`Ctrl + G` を押します。Tツールでは対象面を最初に左クリックします。この最初のクリックは起点設定と準備開始だけで、点追加・適用には再利用されません。
2. カーソル下の可視面を起点に、同じ Face Set ID の同一 edge-connected component を取得します。準備中は工程と進捗を表示し、`Esc` または右クリックで安全に取消できます。
3. 準備完了後、同じ成分上を LMB でクリックしてガイド点を追加します。点の間は表面へ投影した滑らかな曲線として表示されます。
4. `Backspace` で直前の点を戻せます。Wheelは左右共通のplaning半幅を拡大し、Wheel Downは縮小します。`Shift + Wheel`は細かく調整します。HUDとrail previewで幅を確認し、`Enter`で計算・適用を開始します。幅変更は同じguide frameを再利用し、軽いrail更新だけを行います。
5. `Esc` または右クリックは、入力中のガイドを破棄してメッシュを変更しません。
6. 適用後は Blender 標準の `Shift + R` (Repeat Last) を使えます。保存済みguideと幅を現在の同じsurfaceへ再投影し、同じplaning方向で独立したUndoステップとして適用します。

Face Set の境界、非表示面・非表示頂点、完全マスク頂点、同じ Face Set ID でも非連結の成分、対象外オブジェクトは変更しません。部分マスク頂点はマスク量に応じて変形量を減衰させます。共有 Mesh、Shape Key、Multires、Dyntopo、リンク／評価済み Mesh、ライブラリオーバーライドなど、安定した頂点対応を保証できない状態では開始またはRepeatを拒否します。ガイド、対象オブジェクト、Mesh datablock、可視性、トポロジーが無効になった場合も、変更せず安全に取消します。

## Smart Fill

`E` は Sculpt Mode のカーソル下の可視面を seed に、geometry-only の局所解析を開始します。通常 E と `Ctrl + E` は、メッシュの接続、法線、曲率、実距離、hidden / non-manifold 境界、マスクを判定材料にします。ビューポートの明度、照明、matcap、Face Set の表示色、画面キャプチャは候補判定に使いません。

Vertex Paint では同じカーソル面からアクティブな FLOAT_COLOR/BYTE_COLOR の色を三角形の補間（判定不能な n-gon は面内平均）でサンプルします。CORNER 属性は対象面の loop だけ、POINT 属性は対象面が参照する共有頂点へサンプル色を書きます。対応属性がない場合やプレビュー中に属性・seed色が変わった場合は警告して無変更で終了します。

通常 E は境界を穏やかに補正し、`Ctrl + E` は geometry-strict な境界だけを採用します。開始 E の release 後に E を再押下するか `Enter` で一度だけ確定します。Sculpt ModeではFace Set属性へ、Vertex Paintではサンプル色をアクティブ色属性へ書き込みます。予測中は外周境界線だけを描画し、実メッシュ座標・色は変更しません。距離をホイールで変更でき、`Esc` で取消できます。

確定時は Ready プレビューに表示された全 candidate face をそのまま対象にします。Sculpt Modeでは表示済みの全島を一度にFace Setへ書き込み、Vertex PaintではCORNERなら各face loop、POINTなら参照頂点へサンプル色を書き込みます。

Smart Fill 自身の確定直後は surface adjacency / cursor cache を再利用し、座標・可視性・トポロジー・変換などの更新時は安全側に無効化します。

`Shift + Alt + E` はこのソルバーとは独立した表示補助です。現在の View3D に toon-dark MATCAP、Face Set 色、wire overlay を表示し、終了時・ファイルロード時・アドオン解除時に表示設定を復元します。表示補助の切替は通常 E の候補や結果を変更しません。

非表示面、非多様体 edge、未接続 seam、別 sheet、メッシュ境界を越える候補は保護します。大規模メッシュでは局所 graph の準備に時間がかかることがあります。形状上区別できない境界や有効な連結領域がない場合は、無変更で終了します。

## Curved Face Set Tube Shape

`Ctrl + Alt + T` は、カーソル下の seed Face Set の edge-connected component を解析します。局所断面と曲がった中心線を使い、円形だけでなく楕円形・扁平形・軽い不規則形状も扱います。外部境界が二つで均一化を支持する場合はモード A、一つの閉じた尖りと単調な taper を確認できる場合だけモード B の先細りを予測します。

予測中は実メッシュを変更しません。確定時には、対象成分内の安全な頂点だけへ適用します。端部、Face Set 外、境界共有頂点、hidden、完全 mask は固定し、部分 mask は重みで減衰します。分岐、open edge、non-manifold edge、曖昧な tip や断面は理由を表示して拒否します。`Esc`、右クリック、モード・オブジェクト・トポロジー・可視性の変更、Undo、ファイルロード、アドオン解除では cleanup します。

## Topology Colors

Edit Mode で面を選択し、`Ctrl + Alt + 1`〜`6` を押すと、表示中の選択面へ色番号を保存して半透明ガイドを表示します。`Ctrl + Alt + 0` は色を解除します。色はマテリアルではなく、active Edit Mesh の FACE 整数属性 `mfo_topology_color` (0=解除、1〜6=色) へ保存され、.blend と Undo/Redo に含まれます。

非表示面、未選択面、別オブジェクト、既存マテリアルは変更しません。`MFO > Topology Colors` から表示の ON/OFF、透明度、割当、解除を操作できます。

## Preferences

`Edit > Preferences > Add-ons > Mesh Focus Orbit` にあります。

- `Enable`: アドオン全体の有効／無効
- `Activation Key`: 通常 MFO と FSMFO のダブルタップキー
- `Reference Object`: FSMFO が中央レイと Face Set を読む参照メッシュ
- `Focus Loss Behavior`: Blender ウィンドウのフォーカス喪失時の動作
- `Double-tap Window`: ダブルタップ判定の時間幅
- `Show Mode Indicator`: MFO / FSMFO の状態表示
- `Debug Display`: Orbit 中心のデバッグ表示
- `RetopoFlow Focus-Island Snap/Weld Filter`: FSMFO 中の RetopoFlow 候補制限。初期値 OFF
- `Topology Colors` / `Topology Color Opacity`: Topology Colors の表示と透明度

## 制限と復旧

- Guided Ridge と Tube Shape は Sculpt Mode 専用です。Smart Fill は Sculpt Mode と Vertex Paint に対応し、Topology Colors は Edit Mode 専用です。
- Vertex Paint の Smart Fill は、評価済み面と元色属性の対応を保証できないため、ビューポートで有効なモディファイア／変形ジオメトリと非Basis Shape Keyを安全に拒否します。モディファイアなしの元Meshでは使用できます。
- 通常 MFO は Reference Object がなくても、中央レイが表示中の MESH に当たれば起動します。
- FSMFO は Reference Object、`.sculpt_face_set`、中央レイの Face Set hit が必要です。
- RetopoFlow 連携は RetopoFlow がインストールされている場合だけ有効です。
- Shape Key、共有 Mesh、Multires、Dyntopo、リンク Mesh、評価済み MeshなどはGuided Ridgeの直接書き込み対象外です。
- ファイルロード、Undo による対象変更、モード・オブジェクト・Mesh・可視性・トポロジーの変更、アドオン無効化では一時状態・予測・draw handler・isolation を cleanup します。
- Blender標準のUndo/Redoは初回適用と各Repeatを個別に戻せます。保存済みguideの対象が無効になった場合はRepeatを適用しません。

---

# Mesh Focus Orbit — English

A Blender 5.2 add-on for manual retopology workflows.

Add-on version: **3.3.36**

## Main features

- **Normal MFO**: in Object, Edit, or Sculpt Mode, temporarily orbits around the first visible MESH surface hit by the center ray of the current 3D Viewport.
- **Face Set MFO (FSMFO)**: temporarily shows the Face Set hit on the configured Reference Object and isolates a clearly matching Retopo island when possible.
- **Guided Ridge**: fixes a guide crest and user-selected shared width rails, then planes only existing protrusions inward inside that width. It does not fill depressions or move the guide, anchors, backside, or protected vertices.
- **Smart Fill**: in Sculpt Mode it previews a local Face Set region; in Vertex Paint it samples the active FLOAT_COLOR/BYTE_COLOR POINT/CORNER attribute under the cursor and writes that sampled color only on confirm.
- **Shadow Analysis View**: toggles an independent analysis display in the current View3D with `Shift + Alt + E`.
- **Topology Colors**: stores six translucent topology guide colors on selected Edit Mode faces.
- **Curved Face Set Tube Shape**: equalizes or tapers a curved Face Set tube while following its local cross-section.

## Install

1. Open `Edit > Preferences > Add-ons > Install...`.
2. Select `mesh_focus_orbit.py`.
3. Copy the distribution's `assets/mfo-toolbar-icons-astra/` beside the add-on using the same relative path (missing files safely fall back to Blender's standard icons).
4. Enable `Mesh Focus Orbit`.
5. For FSMFO, set a Face Set-bearing reference mesh in Add-on Preferences > `Reference Object`.

The add-on does not automatically save a scene. `Reference Object` is used by FSMFO only. Normal MFO ray-casts the visible MESH objects in the current View3D and uses the nearest hit; Guided Ridge operates on the current active MESH.

The T-toolbar uses five custom Blender VCO `.dat` geometry icons generated from the editable Astra Blender source in `work/astra-icon-modeling/`. Keep those `.dat` files beside the add-on under `assets/mfo-toolbar-icons-astra/`; every tool falls back to a shipped Blender icon when its custom file is unavailable.

### T-toolbar and N-sidebar

The left T-toolbar contains resident, mode-specific MFO tools. Select `MFO: Focus Surface` and left-click a mesh surface to use that click location as the normal MFO target. In Object/Edit Mode, `Ctrl + left-click` on the same tool follows the existing strict Face Set MFO activation path. Dedicated Face Set MFO, Smart Fill (Ctrl for strict), Guided Ridge, and Tube Shape tools appear only in their supported modes. Smart Fill uses Face Sets in Sculpt Mode and the active color attribute in Vertex Paint.

The `MFO` tab in the right N-sidebar exposes `Reference Object` and compact daily settings without opening Preferences. It edits the same Scene property used by Preferences, so there is no duplicated source of truth. Clicks always use the event's 3D View WINDOW-region coordinates rather than the viewport center or a previous brush position.

## Basic operations

| Feature | Mode | Start / confirm | Cancel / notes |
| --- | --- | --- | --- |
| Normal MFO | Object / Edit / Sculpt | Press the configured Activation Key twice quickly | Press it twice again to leave |
| Face Set MFO | Object / Edit | `Ctrl` + Activation Key twice quickly | Press the same combination again to leave |
| MFO T-tool | Object / Edit / Sculpt | Select in T-toolbar, then left-click a surface (Ctrl uses Face Set in Object/Edit) | Re-click the same entry inside the 3D View region to toggle OFF; while Face Set is ON, use the Face Set entry |
| Smart Fill T-tool | Sculpt / Vertex Paint | Select in T-toolbar, then left-click (Ctrl for strict) | Release the start click, then use the next LMB to confirm; existing Smart Fill modal controls apply afterward |
| Guided Ridge / Tube Shape T-tools | Sculpt | Select in T-toolbar, then left-click a target surface (the first Guided Ridge click only sets the start point) | Guided Ridge shows preparation/computation progress; `Esc`/right-click cancels. After Ready, LMB adds points, Wheel/Shift+Wheel adjusts the shared requested maximum width, and `Enter` applies |
| Guided Ridge | Sculpt | `Ctrl + G`; the cursor hit starts the guide, LMB adds points, `Enter` applies | `Backspace` removes the last point; `Esc` or right click cancels |
| Guided Ridge Repeat Last | Sculpt | Blender's standard `Shift + R` | Reprojects the saved guide/width and applies an independent Undo step; no Guided Ridge Shift+R keymap is added |
| Smart Fill | Sculpt / Vertex Paint | `E`, release it, then press `E` again or `Enter` | Sculpt uses Face Sets; Vertex Paint uses the active color attribute. Wheel changes distance; `Esc` cancels |
| Strict Smart Fill | Sculpt / Vertex Paint | `Ctrl + E`, release it, then press `E` again or `Enter` | Sculpt uses Face Sets; Vertex Paint uses the active color attribute. Wheel changes distance; `Esc` cancels |
| Shadow Analysis View | Sculpt | `Shift + Alt + E` | Display helper for the current View3D |
| Topology Colors | Edit | `Ctrl + Alt + 1`–`6` | `Ctrl + Alt + 0` clears selected faces |
| Curved Face Set Tube Shape | Sculpt | `Ctrl + Alt + T`; confirm with `T`, LMB, or `Enter` | `Esc` or right click cancels; wheel changes radius/taper, `Shift + wheel` changes correction strength |

Every modal feature watches its starting object, mesh, mode, visibility, and topology. An external change ends the session without applying it.

## Normal MFO

The default Activation Key is `Right Shift`. Normal MFO ray-casts once from the center of the 3D Viewport and uses the nearest visible MESH surface as a temporary orbit center. Rotate with the Navigation Gizmo or MMB.

When the session ends, the starting `view_location`, `view_distance`, `view_rotation`, and `view_perspective` are restored. Meshes, selection, the 3D Cursor, Pivot Point, object transforms, and the persistent Orbit Around Selection setting are not changed.

## Face Set MFO

FSMFO reads `.sculpt_face_set` from the Reference Object. It displays the hit Face Set through a temporary proxy and treats each edge-connected Retopo component as an island. The island is isolated only when its distance and coverage match are unambiguous; otherwise Retopo isolation is left unchanged.

The Reference Object is not made active and Retopo_Work remains in Edit Mode. The starting visibility and isolation are restored on exit, while newly created geometry keeps its current visibility.

When `RetopoFlow Focus-Island Snap/Weld Filter` is enabled, PolyPen Snap and Translate Auto Merge candidates during FSMFO are limited to existing vertices confirmed to belong to the recorded target island. It is inactive without RetopoFlow and outside FSMFO.

## Guided Ridge

This section describes Guided Ridge behavior in version 3.3.36.

Guided Ridge (3.3.36) fixes the guide as a crest line and builds two planing target planes from the shared requested maximum half-width shown by the rails. Where the surface narrows, each side tapers independently to the available surface without jumping to another sheet; preview and apply use the same projected rails. It moves only existing protrusions inside the selected width inward toward those planes; it never adds material or fills depressions. Guide, rail anchors/support, width outside, backside, Face Set boundaries, hidden vertices, and fully masked vertices remain fixed, while partial masks attenuate the displacement. Preview rails, candidate target planes, fixed-anchor checks, and Repeat share the same captured width/frame result. The candidate uses each projected rail's actual local q/h endpoint rather than nominal width, and preserves triangle/edge barycentric support so every rail support remains fixed. Projection gaps and clearly competing nearby sheets are rejected without a mesh write. Preparation and post-Enter computation yield through timer slices with progress HUD and `Esc`/right-click cancellation. Guided Ridge may start while Normal/Face Set MFO remains active; viewport MMB/NDOF/NUMPAD/Navigation Gizmo events pass through during guide editing.

1. In Sculpt Mode, place the cursor over a Face Set and press `Ctrl + G`. With the T-tool, left-click the target surface first. That first click only sets the start point and begins preparation; it is never reused as a point or confirmation.
2. The visible cursor hit becomes the start point, and the add-on captures the same Face Set ID's single edge-connected component. Preparation shows its stage/progress; `Esc` or right-click cancels safely.
3. After the Ready state, click LMB on that component to add guide points. The guide is displayed as a smooth curve projected onto the surface.
4. Press `Backspace` to remove the last point. Use Wheel to change the shared left/right half-width and `Shift + Wheel` for fine adjustment; the HUD and projected rails show the active width. Width changes reuse the same guide frame and only refresh the lightweight rails. `Enter` computes and applies the planing candidate as one Undo step; `Esc`/right click cancels it.
5. `Esc` or right click discards the in-progress guide without changing the mesh.
6. After confirmation, use Blender's standard `Shift + R` Repeat Last. The saved guide and half-width are reprojected to the same surface and applied with the same planing direction in an independent Undo step.

Face Set boundaries, hidden faces and vertices, fully masked vertices, disconnected components with the same Face Set ID, and other objects are not changed. Partial masks attenuate the displacement according to mask value. Shared Mesh data, Shape Keys, Multires, Dyntopo, linked/evaluated meshes, and library overrides are rejected because stable vertex mapping cannot be guaranteed. If the guide, target object, Mesh datablock, visibility, or topology becomes invalid, the operation cancels without a write.

## Smart Fill

`E` starts a local geometry-only analysis from the visible cursor hit in Sculpt Mode. Normal E and `Ctrl + E` use connectivity, normals, curvature, physical distance, hidden/non-manifold boundaries, and masks. Viewport brightness, lighting, matcap, Face Set display colors, and screen captures are not solver inputs.

In Vertex Paint, the same cursor hit samples the active FLOAT_COLOR/BYTE_COLOR attribute using triangle interpolation (with a face-average fallback for unresolvable n-gons). CORNER attributes write only the candidate face loops; POINT attributes write the referenced shared vertices, preserving Blender's shared-vertex interpolation semantics. Missing attributes or a changed attribute/seed color during preview are reported and leave the mesh unchanged.

Normal E uses a tolerant boundary correction; `Ctrl + E` keeps a geometry-strict boundary. Release the starting E, then press E again or `Enter` to confirm once. Sculpt Mode writes the candidate Face Set; Vertex Paint writes the sampled color through its active CORNER/POINT attribute. Only the outer boundary is drawn during prediction; the mesh coordinates and colors are unchanged until confirmation. The wheel changes the distance and `Esc` cancels.

On confirmation, every candidate face shown by the Ready preview is written as-is. If the candidate contains multiple disconnected islands, all displayed islands are applied; confirmation does not re-limit the result to the seed-connected island.

Immediately after a Smart Fill confirmation, the surface adjacency/cursor caches are reused; coordinate, visibility, topology, transform, and other unowned updates invalidate them conservatively.

`Shift + Alt + E` is an independent display helper. It shows a toon-dark MATCAP, Face Set colors, and a wire overlay in the current View3D, then restores the display settings on exit, file load, or add-on unload. It does not change the normal E solver or its result.

Hidden faces, non-manifold edges, disconnected seams, separate sheets, and mesh boundaries are protected. Initial local graph preparation can take time on large meshes. Vertex Paint Smart Fill currently rejects enabled viewport modifiers/deform geometry and non-basis shape keys because evaluated hit triangles do not yet have a guaranteed source-color mapping; unmodified meshes remain supported. Ambiguous boundaries and regions without a valid connected area are rejected without a write.

## Curved Face Set Tube Shape

`Ctrl + Alt + T` analyzes the edge-connected component of the cursor's seed Face Set. It follows a curved centerline and the local cross-section, including elliptical, flattened, and mildly irregular profiles. Two external boundaries can select uniform mode A; tapered mode B is selected only when one closed pointed tip and monotone taper are supported.

Prediction does not write the mesh. Confirmation writes only safe vertices in the component. End points, vertices outside the Face Set, shared boundary vertices, hidden vertices, and fully masked vertices remain fixed; partial masks reduce the movement. Branches, open edges, non-manifold edges, and ambiguous tips or sections are rejected. `Esc`, right click, mode/object/topology/visibility changes, Undo, file load, and add-on unload clean up the session.

## Topology Colors

In Edit Mode, select faces and press `Ctrl + Alt + 1`–`6` to store a color number and show a translucent guide. `Ctrl + Alt + 0` clears it. The value is stored as the FACE integer attribute `mfo_topology_color` (0=clear, 1–6=color) on the active Edit Mesh, not as a material, and is included in .blend and Undo/Redo.

Hidden faces, unselected faces, other objects, and existing materials are not changed. Use `MFO > Topology Colors` to toggle visibility, set opacity, assign, or clear colors.

## Preferences

Open `Edit > Preferences > Add-ons > Mesh Focus Orbit`.

- `Enable`: enable or disable the add-on
- `Activation Key`: the Normal MFO and FSMFO double-tap key
- `Reference Object`: the mesh whose center ray and Face Set are read by FSMFO
- `Focus Loss Behavior`: behavior when the Blender window loses focus
- `Double-tap Window`: the double-tap time window
- `Show Mode Indicator`: MFO / FSMFO status text
- `Debug Display`: display the orbit-center debug point
- `RetopoFlow Focus-Island Snap/Weld Filter`: FSMFO RetopoFlow candidate restriction; off by default
- `Topology Colors` / `Topology Color Opacity`: Topology Colors visibility and opacity

## Limits and recovery

- Guided Ridge and Tube Shape are Sculpt Mode features. Smart Fill supports Sculpt Mode and Vertex Paint; Topology Colors is an Edit Mode feature.
- Vertex Paint Smart Fill safely rejects enabled viewport modifiers/deformation geometry and non-basis Shape Keys until evaluated-face to source-color mapping is implemented. Unmodified source meshes are supported.
- Normal MFO works without a Reference Object when the center ray hits a visible MESH.
- FSMFO requires a Reference Object, `.sculpt_face_set`, and a center-ray Face Set hit.
- RetopoFlow integration is active only when RetopoFlow is installed.
- Shape Keys, shared Mesh data, Multires, Dyntopo, linked/evaluated meshes, and library overrides are not direct-write targets for Guided Ridge.
- File load, Undo-driven target changes, mode/object/mesh/visibility/topology changes, and add-on disable clean up temporary state, predictions, draw handlers, and isolation.
- Blender's standard Undo/Redo can undo and redo the initial application and each Repeat independently. If the saved guide target is invalid, Repeat applies no changes.
