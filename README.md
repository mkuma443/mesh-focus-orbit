# Mesh Focus Orbit

Current source version: 3.4.12.

Blender 5.2 用のリトポロジー支援アドオンです。

アドオンバージョン: **3.4.12**

## 主な機能

- **通常 MFO**: Object / Edit / Sculpt Mode で、現在の 3D Viewport の中央レイが最初に当たる表示中の MESH 面を一時的な Orbit 中心にします。
- **Face Set MFO (FSMFO)**: Reference Object の中央レイが当たる Face Set と、対応が明確な Retopo island だけを一時表示します。
- **Guided Ridge Curve Sculpt (Step 2)**: Sculpt Modeで各クリックを可視mesh面へスナップした3Dルートを滑らかなカーブとしてプレビューし、明示的なEnterでBlender 5.2 ESSENTIALSのPinch/Magnify、Ctrl+EnterでCrease Polishをbrush.asset_activate経由で選択し、表示中の曲線を同期sculpt.brush_strokeで一度適用します。適用前はメッシュを変更せず、完了後は元のブラシ資産とツールを復元します。Dyntopo中はBlenderネイティブUndo/BMLogを壊さないため適用せず、警告して無変更で戻ります。
- **Smart Fill**: Sculpt Mode では Face Set、Vertex Paint ではアクティブな FLOAT_COLOR/BYTE_COLOR の POINT/CORNER 属性を使って、カーソル下の局所領域をプレビューし、確定時に同じシード色を書き込みます。`Shift + E` は隣接面の法線角だけを使う軽量モードで、明確な山・谷を越える距離コストを平面の2倍にして拡大します。
- **Wire Overlay View**: どのモードでも `Shift + Alt + E` で対象 View3D のワイヤー表示を現在値から切り替えます。
- **表示距離集中**: `Shift + Alt + V` で対象 View3D の遠方クリップを 0.13 m と 1000 m の間で切り替えます。
- **Opening Boundary Loop**: Edit Modeで開口部の辺を1本選択し、`Shift + Alt + L`で同じ開口部の閉じた縁を一周選択します。
- **Topology Colors**: Edit Mode の選択面に 1〜6 の半透明ガイド色を保存します。
- **Curved Face Set Tube Shape**: 曲がった Face Set チューブを局所断面に沿って均一化または先細り補正します。

## インストール

1. `Edit > Preferences > Add-ons > Install...` を開きます。
2. `mesh_focus_orbit/__init__.py` を含む配布zipを選択するか、同パッケージディレクトリ全体をアドオン配置先へコピーします。単一ファイルへ平坦化しないでください。
3. 配布物の `assets/mfo-toolbar-icons-astra/` を、アドオンファイルと同じ配置先の相対 `assets/mfo-toolbar-icons-astra/` へコピーします（欠落時はBlender標準アイコンへフォールバックします）。
4. `Mesh Focus Orbit` を有効にします。
5. FSMFO を使う場合は、Add-on Preferences の `Reference Object` に Face Set を持つ参照メッシュを指定します。

アドオンはユーザー設定以外のシーンを自動保存しません。Reference Object は FSMFO 専用です。通常 MFO は現在の View3D で表示中の MESH 群を中央レイで調べ、最前面の hit を対象にします。Guided Ridge は現在のアクティブな MESH を対象にします。

Tツールバーは、編集可能なAstra Blenderソースから生成した5種のBlender VCO `.dat` ジオメトリアイコンを使用します。`assets/mfo-toolbar-icons-astra/` をアドオンの相対パスへ保持してください。欠落時は機能ごとにBlender標準アイコンへ安全にフォールバックします。

内部実装は `mesh_focus_orbit/` パッケージへ分割されています。`foundation.py` が共有のBlender依存・基礎機能、`guided_ridge/core.py` がGuided Ridge、`local_feature.py` と `tube_shape.py` が補助機能、`smart_fill/geometry.py` と `smart_fill/preview.py` がSmart Fill/Vertex Paint、`registration.py` が登録・ツール・アイコンを担当します。`config.py` は不変設定、`runtime.py` は共有実行時状態、`lifecycle.py` は明示的な登録解除とcallback identity cleanupを担当します。`__init__.py` は依存順loaderと公開登録入口だけを持ち、既存のoperator ID・keymap・互換参照を意図的にre-exportします。配布時は単一pyへ平坦化せず、パッケージ内の全モジュールを含めてください。

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
| Guided Ridge / Tube Shape Tツール | Sculpt | ツールで選択後、対象面を左クリック（Guided Ridgeの最初のクリックは起点設定のみ） | Guided Ridgeは準備進捗を表示。準備完了後にLMBで点追加、`Enter`でCurve Previewへ。Preview中の`Enter`はPinch、`Ctrl+Enter`はCreaseを適用 |
| Guided Ridge | Sculpt | `Ctrl + G`。カーソル下の面を起点にし、LMBで点を追加、`Enter`でCurve Previewへ | Preview中はWheel/Shift+Wheelで平滑化、LMBはルート編集へ漏れず、`Backspace`で編集へ戻り、`Esc`/右クリックで取消。`Tab`/Scrape仕上げは後続Stepです |
| Guided Ridge Repeat Last | Sculpt | Blender標準の `Shift + R` | 既存のRepeat Last互換経路。Curve Sculptの反復はPreview中に現在の曲線で`Enter`/`Ctrl+Enter`を使用します。標準Undo境界はBlender側が所有します |
| Smart Fill | Sculpt / Vertex Paint | `E`、開始キーを離してから `E` または `Enter` | SculptはFace Set、Vertex Paintはアクティブ色属性。ホイールで距離、`Esc` で取消 |
| Smart Fill Expand Only | Sculpt / Vertex Paint | `Shift + E`、開始キーを離してから `E` または `Enter` | サーフェス／valley／ridge／edge-cost評価を省略し、トポロジー距離だけで拡大。ホイールで段階変更、`Esc` で取消 |
| Strict Smart Fill | Sculpt / Vertex Paint | `Ctrl + E`、開始キーを離してから `E` または `Enter` | SculptはFace Set、Vertex Paintはアクティブ色属性。ホイールで距離、`Esc` で取消 |
| Wire Overlay View | 3D View（全モード） | `Shift + Alt + E` | 対象 View3D の現在の wire 状態を反転。ON時は薄い表示、OFF時は他のoverlay設定を変更しない |
| 表示距離集中 | 3D View（全モード） | `Shift + Alt + V` | 初回は0.13 mで調整開始。ホイール調整後にLMB/Enterで確定し、作業後に再押下で1000 mへ復元。Esc/右クリックは1000 mへ取消。保存ファイルは1000 mで保存し、完了/失敗後は現在画面の調整値へ戻します |
| Opening Boundary Loop | Edit（辺選択） | 開口部の辺を1本または2本選択して `Shift + Alt + L` | 1本なら一周、2本なら短区間。選択を変えず再実行すると反対区間へ交互に切替 |
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

この節は3.4.1のGuided Ridge Curve Sculpt Step 2動作を説明します。

負側/初期値のinscribed表示は履歴上の3.3.53構築をそのまま用い、正側Amplified専用のlobe正規化・waveform validatorは通しません。負側はその履歴BezierのC1/端点を保持し、正側の例外や計算量から独立しています。

Guided Ridge Curve Sculpt（3.4.1）は、可視mesh面へスナップした3Dルートを現在ビューの2Dへ再投影し、弧長基準のC1 Bezier波形として非破壊プレビューします。明示的なEnterでESSENTIALSのPinch/Magnify、Ctrl+EnterでCrease Polishを同期sculpt.brush_strokeで実行します。適用前のプレビューではメッシュを変更せず、各適用後に元のブラシ資産とツールを復元します。BlenderがRUNNING_MODAL/PASS_THROUGHを返す場合は同期適用未完了として警告し、変更を成功扱いにしません。Dyntopo中はネイティブBMLog/Undoの破損を避けるため、資産切替やstrokeを開始せず警告して無変更で戻ります。Shape=0はraw折れ線、正側Amplified（強調）はraw基準の波形増幅、負側Attenuated（減衰）は履歴inscribed構築です。

平滑化ターゲットとBezier接線は現在ビューの2Dで計算し、元の3D点列はビュー変更時の再投影にだけ使います。適応サンプル数は画面長に応じて64〜512点に制限し、C1接線連続、端点固定、候補ごとのスケール相対的な進行検証を適用します。Shape=0だけが手動折れ線をそのまま表示し、非ゼロ値はBezier生成された滑らかな表示になります。正側は現行のAmplified waveform（局所振幅を増幅）、負側と初期値は履歴inscribed構築（局所振幅を減衰）として分離され、正側のS字・複数ローブ処理が負側の復元ジオメトリを変更しません。直線ルートはノット密度によらず全Shape値で直線のままです（テストは代表的な平面経路の進行・折返しを検証し、一般3D自己交差は保証しません）。

非ゼロ曲線は最大512点で描画・検証します。同じルート・Shape・投影行列・領域サイズ/UI倍率・オブジェクト/メッシュ識別子ではTIMERは文脈確認だけを行い、再フィットや交差検証を繰り返しません。ビューまたはルート変更時だけ一度再生成します。候補失敗時は同一ビューなら直前の有効表示を保持し、別ビューでは古い画面座標を表示せず現在ビューのrawと警告だけを表示します。現在ビューでraw点の投影自体は完了しているものの領域外の点がある場合も、投影済みrawを保持して警告します。投影が同一点へ縮退した場合もraw投影点と明示的な警告を保持し、成功した直線曲線とは扱いません。空のscreen配列を有効なcache hitとは扱いません。

1. Sculpt Mode で表示中のmesh surfaceにカーソルを置き、`Ctrl + G` を押します。Tツールでは対象面を最初に左クリックします。この最初のクリックは起点設定だけで、点追加・適用には再利用されません。Face Setの有無や面数はStep 2の開始条件ではありません。
2. カーソル下の可視面を起点に、表面へスナップしたルート編集へ直ちに入ります。大きなFace SetやFace Set未設定のmeshでも、surface hitが得られる限り高コストな旧Face Set準備を行わず開始します。`Esc` または右クリックで安全に取消できます。
3. 準備完了後、可視mesh面を LMB でクリックして、各クリック位置を表面へスナップした3Dガイド点として追加します。点列を現在のビューへ投影し、2D弧長再サンプリングとソフトBezier近似によるプレビューを表示します。クリック間を連続的にsurfaceへ再投影する保証はありません。
4. `Enter` でルート編集から `Curve Preview` へ移ります。HUDには `GUIDED RIDGE - CURVE SCULPT / PREVIEW`、Curve Shape（-100..+100: 負=Attenuated（局所振幅を減衰） / 0=Raw / 正=Amplified（局所振幅を増幅））、操作説明が表示されます。Preview中の `Wheel` は符号付きゲインを粗く調整し、`Shift + Wheel` は細かく調整します。前回の符号付き値は次のセッションでも使われます。
5. Preview中の `Backspace` は点列とPreview開始前のsurface guide/rail cacheをそのまま保持してルート編集へ戻ります。`Enter` は専用Pinch Ridge、`Ctrl + Enter` は専用Crease Polish Valleyを現在の表示曲線へ一度適用します。反復入力は同じ曲線を再実行します。`Esc` または右クリックは未適用なら一時表示を片付けてメッシュを変更せず終了し、適用後はブラシ/一時native curve resourcesを片付けて結果をBlender標準Undoへ委ねます。
6. ビュー回転・パン・ズーム・Navigation Gizmo・領域サイズ変更は既存の透過経路を維持します。保持した3Dルートから表示を更新し、現ビューへ全点を投影できない場合は警告して変更しません。モード・オブジェクト・ファイル・アドオン状態の変更時も一時ハンドラとタイマーを解除します。

Shape 0 から ±1 への切替は、raw折れ線からC1 Bezier族へ入る意図的なモード切替です。HUDでは正側をAmplified（強調）、負側をAttenuated（減衰）と表示します。±1以降の非ゼロ値は全て滑らかなBezier表示で、各波形ローブの振幅を局所的に増減します。

3.3.70では新規ルートの初期Shapeを-12（減衰側）とし、outward waveform導入前の3.3.53で検証した現在ビュー2Dのinscribed C1構築を復元して最初に表示します。負側は同じ歴史的な弧長・ロバスト近傍平滑化・端点固定Bezier構築を使い、正側のAmplified変換とは独立です。Shape 0はraw折れ線、Wheelで明示的にShapeを変更できます。ローブ区間はcleaned曲線ではなくrawノットの弧長・符号ピーク・谷から決めます。振幅の最終判定は実際に描画する最大512点のpolylineと区間境界の線形補間だけを使い、非表示のcontrol peakでは成功扱いにしません。

## Smart Fill

`E` は Sculpt Mode のカーソル下の可視面を seed に、geometry-only の局所解析を開始します。通常 E と `Ctrl + E` は、メッシュの接続、法線、曲率、実距離、hidden / non-manifold 境界、マスクを判定材料にします。ビューポートの明度、照明、matcap、Face Set の表示色、画面キャプチャは候補判定に使いません。

1回のSmart Fillプレビューでは、可視面の集合とseedから到達できる可視componentを開始時に固定します。Hidden面は安定した障壁として扱い、候補のサンプリングや再導入には使いません。通常モードの拡大は直前に受理した面を必ず保持する単調な上位集合になり、縮小は保存済みの該当radius結果をそのまま復元します。プレビュー中に可視性またはトポロジーが変わった場合は、古い集合を黙って混ぜずにセッションを取消して再準備します。

Vertex Paint では同じカーソル面からアクティブな FLOAT_COLOR/BYTE_COLOR の色を三角形の補間（判定不能な n-gon は面内平均）でサンプルします。CORNER 属性は対象面の loop だけ、POINT 属性は対象面が参照する共有頂点へサンプル色を書きます。対応属性がない場合やプレビュー中に属性・seed色が変わった場合は警告して無変更で終了します。

通常 E は境界を穏やかに補正し、`Ctrl + E` は geometry-strict な境界だけを採用します。開始 E の release 後に E を再押下するか `Enter` で一度だけ確定します。Sculpt ModeではFace Set属性へ、Vertex Paintではサンプル色をアクティブ色属性へ書き込みます。予測中は外周境界線だけを描画し、実メッシュ座標・色は変更しません。距離をホイールで変更でき、`Esc` で取消できます。

通常Smart Fillは、現在の境界が描画されて確定可能になった後のWheelを1回につき1段だけ受理します。準備中・計算中・描画待ち・短い再開待ちに届いた追加Wheelはキューせず破棄するため、速く回しても遅く回しても同じ段階列になります。初期半径の約8.192倍（最大半径の1/1.25^3）以降の単調な拡大では、最初の完全結果を端点ベースとして保持し、全メッシュcoverageが証明されたグラフ上で、既存の形状境界を越えずにDijkstra frontierから新しい面だけを追加する終端拡張を使います。さらに選択済み面が10万以上の通常モードでは、表示HUDに`frontier-delta`と示し、前回境界のうち新規面に接する辺だけを更新します（描画・確定用の不変配列を世代ごとに一度だけ作成）。証拠が不足・縮小・トポロジー/可視性変更時は完全計算へ安全に戻り、保持した面を縮小結果へ流用しません。Ctrl+Eの厳密経路は従来どおりです。

確定時は Ready プレビューに表示された全 candidate face をそのまま対象にします。Sculpt Modeでは表示済みの全島を一度にFace Setへ書き込み、Vertex PaintではCORNERなら各face loop、POINTなら参照頂点へサンプル色を書き込みます。

Smart Fill 自身の確定直後は surface adjacency / cursor cache を再利用し、座標・可視性・トポロジー・変換などの更新時は安全側に無効化します。

`Shift + Alt + E` はモードに依存しない表示操作です。対象 View3D の `show_wireframes` を毎回現在値から反転し、ON時は通常の薄い wire 設定を適用します。OFF時は wire 表示だけをOFFにし、他のoverlay設定は復元・変更しません。保存後に再起動しても、その View3D の現在の wire 状態から切り替わります。

`Shift + Alt + V` の初回押下で対象 View3D の遠方クリップ `clip_end` を0.13 mにし、調整中はホイールで距離を変更します。LMBまたはEnterで距離を確定するとモーダルを終了し、通常のホイール操作へ戻ります。作業後に再押下すると1000 mへ復元し、Esc/右クリックは調整を取消して1000 mへ戻します。`clip_start`、視点のズーム距離、他の View3D の値は変更しません。

非表示面、非多様体 edge、未接続 seam、別 sheet、メッシュ境界を越える候補は保護します。大規模メッシュでは局所 graph の準備に時間がかかることがあります。形状上区別できない境界や有効な連結領域がない場合は、無変更で終了します。

## Curved Face Set Tube Shape

`Ctrl + Alt + T` は、カーソル下の seed Face Set の edge-connected component を解析します。局所断面と曲がった中心線を使い、円形だけでなく楕円形・扁平形・軽い不規則形状も扱います。外部境界が二つで均一化を支持する場合はモード A、一つの閉じた尖りと単調な taper を確認できる場合だけモード B の先細りを予測します。

予測中は実メッシュを変更しません。確定時には、対象成分内の安全な頂点だけへ適用します。端部、Face Set 外、境界共有頂点、hidden、完全 mask は固定し、部分 mask は重みで減衰します。分岐、open edge、non-manifold edge、曖昧な tip や断面は理由を表示して拒否します。`Esc`、右クリック、モード・オブジェクト・トポロジー・可視性の変更、Undo、ファイルロード、アドオン解除では cleanup します。

## Topology Colors

`Shift + Alt + L` は、Edit Modeで可視開口部辺を1本選択するとその開口の縁を一周選び、同じ開口の辺を2本選択すると両端の辺を含む2区間のうち幾何学的に短い区間を選びます。2辺モードで選択を変えず再実行すると反対側の区間へ切り替わり、再実行ごとに交互に選択します。Face Setなどで面を非表示にしてできた開口では、表示面と非表示面の境界を追跡し、面が欠けている実境界にも対応します。画面外の辺も含み、視点や面の色には依存しません。別の開口と1つの頂点で接している場合は閉路を分離します。別開口の辺、境界でない辺、曖昧または途切れた閉路、探索上限超過、再実行tokenと選択・対象・局所形状や可視性が一致しない場合は安全に扱い、失敗時は選択を変更しません。Undo/Redo、ファイル読み込み、アドオン再読み込みも再実行状態を破棄します。面の生成、非表示面の再表示、メッシュ形状の変更は行いません。同じ操作は `MFO > Topology Colors > 開口部を一周選択` からも実行できます。

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
- `Guided Ridge Curve Shape` / `Guided Ridge Strength` / `Guided Ridge Radius`: Curve Sculpt の記憶値と専用ブラシ設定

## 制限と復旧

- Guided Ridge と Tube Shape は Sculpt Mode 専用です。Smart Fill は Sculpt Mode と Vertex Paint に対応し、Topology Colors は Edit Mode 専用です。
- Vertex Paint の Smart Fill は、評価済み面と元色属性の対応を保証できないため、ビューポートで有効なモディファイア／変形ジオメトリと非Basis Shape Keyを安全に拒否します。モディファイアなしの元Meshでは使用できます。
- 通常 MFO は Reference Object がなくても、中央レイが表示中の MESH に当たれば起動します。
- FSMFO は Reference Object、`.sculpt_face_set`、中央レイの Face Set hit が必要です。
- RetopoFlow 連携は RetopoFlow がインストールされている場合だけ有効です。
- Guided RidgeのCurve PreviewはMeshへ書き込まず、Enterで初めてESSENTIALS Pinch/Magnifyを同期 sculpt.brush_stroke で1回適用します。Ctrl+EnterはCrease Polishを適用します。適用前のプレビュー、投影失敗、対象変更ではMeshとUndoを変更しません。Smart Fillのモーダル所有者は終端イベントまで保持し、外部変更時は次のイベントでCANCELLEDを返します。終端後も2回のBlenderメインループ猶予を通過するまでreload/installを安全扱いにしません。通常のreload/disableは登録クラスを完全に解除し、ブラシ復元に失敗した場合は復元記録を保持して再試行または警告します。報告されたモード切替境界のネイティブクラッシュ再現は自動GUI試験から除外しており、実機ネイティブ検証は未完了です。
- ファイルロード、Undo による対象変更、モード・オブジェクト・Mesh・可視性・トポロジーの変更、アドオン無効化では一時状態・予測・draw handler・isolation を cleanup します。
- Esc、右クリック、モード・オブジェクト・Mesh変更はブラシ資産参照とツールを復元します。適用前ならMeshを変更せず、適用後は結果を残してBlender標準Undoへ委ねます。

---

# Mesh Focus Orbit — English

A Blender 5.2 add-on for manual retopology workflows.

Add-on version: **3.4.12**

## Main features

- **Normal MFO**: in Object, Edit, or Sculpt Mode, temporarily orbits around the first visible MESH surface hit by the center ray of the current 3D Viewport.
- **Face Set MFO (FSMFO)**: temporarily shows the Face Set hit on the configured Reference Object and isolates a clearly matching Retopo island when possible.
- **Guided Ridge Curve Sculpt (Step 2)**: previews a click-snapped 3D route, then activates the Blender 5.2 ESSENTIALS Pinch/Magnify or Crease Polish brush and applies one synchronous sculpt.brush_stroke from the accepted current-view curve. Before explicit apply it creates no mesh change; the exact prior asset reference and active tool are restored on exit. RUNNING_MODAL/PASS_THROUGH are treated as incomplete.
- **Smart Fill**: in Sculpt Mode it previews a local Face Set region; in Vertex Paint it samples the active FLOAT_COLOR/BYTE_COLOR POINT/CORNER attribute under the cursor and writes that sampled color only on confirm. `Shift + E` uses a lightweight adjacent-normal test and makes a clear ridge or valley cost twice as much to cross as flat adjacency.
- **Wire Overlay View**: in any mode, `Shift + Alt + E` toggles wire display from the current value of that View3D.
- **Display Distance Focus**: `Shift + Alt + V` starts at 0.13 m and captures the wheel only while adjusting. Click or Enter keeps the distance and ends the modal; a later press restores 1000 m. Esc/right click cancels to 1000 m.
- **Opening Boundary Loop**: select one opening-rim edge in Edit Mode and press `Shift + Alt + L` to select its complete closed rim. With two edges on the same opening, the first press selects the shorter inclusive arc; pressing again without changing the selection alternates to the opposite arc.
- **Topology Colors**: stores six translucent topology guide colors on selected Edit Mode faces.
- **Curved Face Set Tube Shape**: equalizes or tapers a curved Face Set tube while following its local cross-section.

## Install

1. Open `Edit > Preferences > Add-ons > Install...`.
2. Select a distribution zip containing `mesh_focus_orbit/__init__.py`, or copy the complete package directory into the add-ons directory; do not flatten it into a single file.
3. Copy the distribution's `assets/mfo-toolbar-icons-astra/` beside the package using the same relative path (missing files safely fall back to Blender's standard icons).
4. Enable `Mesh Focus Orbit`.
5. For FSMFO, set a Face Set-bearing reference mesh in Add-on Preferences > `Reference Object`.

The add-on entry point is `mesh_focus_orbit/__init__.py`. The implementation is split into normally importable `foundation.py`, `guided_ridge/core.py`, `local_feature.py`, `tube_shape.py`, `smart_fill/geometry.py`, `smart_fill/preview.py`, and `registration.py`, with explicit dependency order. `runtime.py` owns shared mutable session/cache/handler/timer state; `lifecycle.py` owns exact callback cleanup; domain modules call those owners by module reference. The facade exports only intentional compatibility names and does not broadcast assignments. Distribute the complete package; do not flatten it into a single Python file.

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
| Guided Ridge / Tube Shape T-tools | Sculpt | Select in T-toolbar, then left-click a target surface (the first Guided Ridge click only sets the start point) | Guided Ridge shows preparation progress; after Ready, LMB adds points and `Enter` opens Curve Preview. In Preview, `Enter` applies Pinch Ridge and `Ctrl + Enter` applies Crease Polish Valley |
| Guided Ridge | Sculpt | `Ctrl + G`; the cursor hit starts the guide, LMB adds points, `Enter` opens Curve Preview | In Preview, Wheel/Shift+Wheel changes shape, LMB is consumed, `Enter` applies Pinch, `Ctrl + Enter` applies Crease, `Backspace` returns to editing, and `Esc`/right click cleans up. Tab/Scrape finishing is a later step |
| Guided Ridge Repeat Last | Sculpt | Blender's standard `Shift + R` | The current milestone keeps the existing Repeat Last compatibility path; each explicit Curve Sculpt apply is undone through Blender's standard Undo, whose boundary is owned by Blender |
| Smart Fill | Sculpt / Vertex Paint | `E`, release it, then press `E` again or `Enter` | Sculpt uses Face Sets; Vertex Paint uses the active color attribute. Wheel changes distance; `Esc` cancels |
| Smart Fill Expand Only | Sculpt / Vertex Paint | `Shift + E`, release it, then press `E` again or `Enter` | Bypasses surface/valley/ridge/edge-cost evaluation and expands by topology distance only. Wheel changes stages; `Esc` cancels |
| Strict Smart Fill | Sculpt / Vertex Paint | `Ctrl + E`, release it, then press `E` again or `Enter` | Sculpt uses Face Sets; Vertex Paint uses the active color attribute. Wheel changes distance; `Esc` cancels |
| Wire Overlay View | 3D View (any mode) | `Shift + Alt + E` | Toggles the current View3D wire state; turning it OFF leaves other overlay settings unchanged |
| Display Distance Focus | 3D View (any mode) | `Shift + Alt + V` | Starts adjustment at 0.13 m; wheel adjusts, LMB/Enter keeps it and ends the modal, later press restores 1000 m; Esc/right click cancels. The file stores 1000 m and the live viewport value is restored after save success/failure |
| Opening Boundary Loop | Edit (Edge Select) | Select one or two opening-rim edges, then `Shift + Alt + L` | One edge selects the full rim; two select the shorter inclusive arc, and an unchanged repeat alternates to the opposite arc |
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

This section describes the Guided Ridge Curve Sculpt Step 2 application path in version 3.4.1.

The initial and negative/inscribed display uses the historical 3.3.53 construction directly; the positive Amplified lobe normalization and waveform validator are not run for it. Its historical Bezier endpoints and C1 construction remain independent of positive-path exceptions and cost.

Guided Ridge Curve Sculpt retains the accepted 3.3.71/3.3.80 current-view 2D curve construction unchanged. Clicks remain retained 3D surface points; the display is an arc-length, C1 Bezier waveform. Shape 0 is the raw polyline, while the historical inscribed/default and separate amplified paths remain independent. The application milestone consumes only the displayed, validated screen curve and never recomputes or changes that fitting algorithm.

The smooth target and Bezier tangents are computed in the current view's 2D projection; the original 3D route is retained for deterministic reprojection after view changes. The display is adaptively tessellated between 64 and 512 samples, with C1 tangent continuity, exact endpoints, and a scale-relative progression check. Shape 0 is the only raw/manual polyline display; every nonzero value stays smooth. Positive values use the current Amplified local-wave path, while the initial and negative/inscribed values use the restored historical construction in an isolated path, so later outward-wave logic cannot alter that proven baseline. S-curves and multiple lobes remain supported by the positive path; a straight route remains straight for every Shape value regardless of knot density. Tests cover progression and foldback on representative planar routes; general 3D self-intersection is not guaranteed.

Nonzero drawing and validation are capped at 512 points. When route, Shape, projection matrices, region dimensions/UI scale, or object/mesh identity are unchanged, TIMER performs only a lightweight signature/context check and skips fitting/intersection validation. A changed view or route gets one refreshed generation. If a candidate fails, the same view may retain the exact last-valid display; after a view change, stale screen coordinates are cleared and only the current raw route plus a warning may be shown. If the current view yields a complete projection but one or more points are outside the region, the newly projected raw route remains visible with a warning. A projection that collapses all route points to one screen location is diagnosed as unavailable rather than labelled a successful straight curve; the raw projected points and warning remain visible. Empty screen arrays are never accepted as a cache hit.

1. In Sculpt Mode, place the cursor over a visible mesh surface and press `Ctrl + G`. With the T-tool, left-click the target surface first. That first click only sets the start point; it is never reused as a point or confirmation. A Face Set attribute or small Face Set is not required to start the route or reach the Step 2 preview.
2. The visible cursor hit becomes the start point and the add-on enters route editing immediately. Large Face Sets and unpartitioned meshes use the read-only surface hit path without the legacy Face Set snapshot; `Esc` or right-click cancels safely.
3. After the Ready state, click LMB on a visible mesh surface to add each guide point as a click-snapped 3D control. The route is projected into the current view and displayed through a 2D arc-length-resampled, soft Bezier approximation; continuous surface reprojection between clicks is not guaranteed.
4. Press `Enter` to switch from route editing to `Curve Preview`. The HUD shows Curve Shape (-100..+100) and the controls. In Preview, `Wheel` adjusts the signed shape in coarse steps and `Shift + Wheel` in fine steps; the last value is remembered for the next session.
5. In Preview, Enter activates the shipped ESSENTIALS Pinch/Magnify brush and applies the current screen curve synchronously with sculpt.brush_stroke; Ctrl + Enter uses Crease Polish. Repeating either key repeats the same current curve. RUNNING_MODAL and PASS_THROUGH are not treated as completed. Each apply uses an explicit View3D override and restores the exact prior asset reference and active tool. Backspace returns to route editing without losing points or the pre-preview surface-guide/rail cache. Esc or right click before apply leaves the mesh unchanged; Blender's standard Undo owns completed stroke boundaries.
6. View orbit/pan/zoom, the Navigation Gizmo, and region resizing keep the existing pass-through behavior. The retained 3D route is used to refresh the display; if all points cannot be projected in the current view, a warning is shown and no new application is allowed. Mode, object, file, and add-on changes restore the prior asset/tool and clean up handlers and timers.

The transition from Shape 0 to ±1 is an intentional mode switch from the raw polyline into the C1 Bezier families. The HUD labels positive values Amplified and negative values Attenuated. Every nonzero value remains smooth and changes local wave amplitude rather than rejecting S-shaped routes.

In 3.3.70, a new route starts at Shape -12 on the attenuated side, using the restored pre-outward (3.3.53) current-view inscribed C1 construction close to the raw envelope. Negative values stay on that isolated historical construction; positive values use the separate amplified waveform path. Shape 0 remains the exact raw polyline and Wheel can change it explicitly. Lobe intervals come from raw-knot arc length, sign peaks, and valleys rather than the cleaned fit. Final amplitude validation uses only the actual bounded draw polyline plus linearly interpolated interval boundaries; hidden control-space peaks cannot make a displayed candidate pass.

## Smart Fill

`E` starts a local geometry-only analysis from the visible cursor hit in Sculpt Mode. Normal E and `Ctrl + E` use connectivity, normals, curvature, physical distance, hidden/non-manifold boundaries, and masks. Viewport brightness, lighting, matcap, Face Set display colors, and screen captures are not solver inputs.

Each Smart Fill preview captures an immutable visible-face universe and the visible component reachable from the seed. Hidden faces remain hard barriers and are never sampled or reintroduced. Normal growth is monotonic: every accepted stage is a superset of the immediately preceding displayed stage. Shrinking starts a fresh bounded computation; it never reuses additive faces from a larger radius. If visibility or topology changes during the session, the preview is cancelled for explicit re-preparation instead of mixing old and new universes.

In Vertex Paint, the same cursor hit samples the active FLOAT_COLOR/BYTE_COLOR attribute using triangle interpolation (with a face-average fallback for unresolvable n-gons). CORNER attributes write only the candidate face loops; POINT attributes write the referenced shared vertices, preserving Blender's shared-vertex interpolation semantics. Missing attributes or a changed attribute/seed color during preview are reported and leave the mesh unchanged.

Normal E uses a tolerant boundary correction; `Ctrl + E` keeps a geometry-strict boundary. Release the starting E, then press E again or `Enter` to confirm once. Sculpt Mode writes the candidate Face Set; Vertex Paint writes the sampled color through its active CORNER/POINT attribute. Only the outer boundary is drawn during prediction; the mesh coordinates and colors are unchanged until confirmation. The wheel changes the distance and `Esc` cancels.

Normal Smart Fill accepts one Wheel step only after the current boundary has been successfully drawn and re-armed. Wheel events delivered during preparation, computation, draw wait, or the short re-arm interval are dropped rather than queued, so rapid and slow input produce the same per-step sequence; confirmation remains blocked until the accepted result is ready and drawn. At the final three monotonic growth stages (starting at about 8.192 times the initial radius), the first complete result is retained as an exact base and new faces are added from its Dijkstra frontier only when full mesh coverage is certified, without crossing recorded shape barriers or rerunning the expensive refinement passes. When a normal selection already contains at least 100,000 faces, the HUD labels the eligible growth as `frontier-delta`; only edges incident to newly reached faces are updated, while the current displayed generation owns one compact accepted full-graph ID array. Missing provenance, incomplete coverage, a shrink, or a topology/visibility change safely falls back to a full compute and never reuses additive faces for the smaller radius. Ctrl+E keeps its existing strict-mode behavior.

On confirmation, every candidate face shown by the Ready preview is written as-is. If the candidate contains multiple disconnected islands, all displayed islands are applied; confirmation does not re-limit the result to the seed-connected island.

Immediately after a Smart Fill confirmation, the surface adjacency/cursor caches are reused; coordinate, visibility, topology, transform, and other unowned updates invalidate them conservatively.

`Shift + Alt + E` works in any mode. Each press inverts the current View3D's `show_wireframes` value; enabling wires applies the usual faint wire settings, while disabling them changes only the wire flag. The current property remains authoritative after saving and reopening a file.

`Shift + Alt + V` starts a short adjustment modal at 0.13 m. The wheel adjusts only `clip_end`; LMB or Enter keeps the selected distance and ends the modal, so normal wheel zoom resumes. Pressing `Shift + Alt + V` after the work restores 1000 m. Esc or right click cancels to 1000 m. It does not change `clip_start`, view distance, or another View3D.

Hidden faces, non-manifold edges, disconnected seams, separate sheets, and mesh boundaries are protected. Initial local graph preparation can take time on large meshes. Vertex Paint Smart Fill currently rejects enabled viewport modifiers/deform geometry and non-basis shape keys because evaluated hit triangles do not yet have a guaranteed source-color mapping; unmodified meshes remain supported. Ambiguous boundaries and regions without a valid connected area are rejected without a write.

## Curved Face Set Tube Shape

`Ctrl + Alt + T` analyzes the edge-connected component of the cursor's seed Face Set. It follows a curved centerline and the local cross-section, including elliptical, flattened, and mildly irregular profiles. Two external boundaries can select uniform mode A; tapered mode B is selected only when one closed pointed tip and monotone taper are supported.

Prediction does not write the mesh. Confirmation writes only safe vertices in the component. End points, vertices outside the Face Set, shared boundary vertices, hidden vertices, and fully masked vertices remain fixed; partial masks reduce the movement. Branches, open edges, non-manifold edges, and ambiguous tips or sections are rejected. `Esc`, right click, mode/object/topology/visibility changes, Undo, file load, and add-on unload clean up the session.

## Topology Colors

`Shift + Alt + L` follows an opening rim in Edit Mode, including borders between visible and hidden faces (such as hidden Face Sets) and true mesh boundaries. One selected visible rim edge selects the complete closed rim. Exactly two edges on the same unique rim select the shorter continuous arc, including both anchor edges; repeating the command without changing selection or local geometry/visibility alternates to the opposite arc. Openings touching at a single vertex are separated, selecting only the cycle containing the seed. Invalid or ambiguous edges, a broken/branched rim, or a search limit leaves selection unchanged. The command only selects edges; it does not create faces, reveal hidden geometry, or modify the mesh shape. It is also available from `MFO > Topology Colors > Select Opening Boundary Loop`.

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
- `Guided Ridge Curve Shape` / `Guided Ridge Strength` / `Guided Ridge Radius`: remembered Curve Sculpt shape and dedicated brush settings

The Topology Colors service is owned by `foundation._topology_color_object`; it is
not a mutable runtime field. Deferred orphan-cleanup timer callback identities are
owned by `runtime` and are explicitly unregistered during disable/reload.

## Limits and recovery

- Root reload inspects the previous package lifecycle even when its registered flag is already false; active owners and retired-modal quiescence remain fail-closed, and a lost grace timer is re-armed only through the Blender timer registry. The native GUI crash-boundary runner has a common explicit opt-in guard and is not launched by default.
- Guided Ridge and Tube Shape are Sculpt Mode features. Smart Fill supports Sculpt Mode and Vertex Paint; Topology Colors is an Edit Mode feature.
- Vertex Paint Smart Fill safely rejects enabled viewport modifiers/deformation geometry and non-basis Shape Keys until evaluated-face to source-color mapping is implemented. Unmodified source meshes are supported.
- Normal MFO works without a Reference Object when the center ray hits a visible MESH.
- FSMFO requires a Reference Object, `.sculpt_face_set`, and a center-ray Face Set hit.
- RetopoFlow integration is active only when RetopoFlow is installed.
- Guided Ridge Curve Preview is non-destructive until an explicit Enter/Ctrl+Enter. Those keys activate only Blender 5.2 ESSENTIALS Pinch/Magnify or Crease Polish through the public brush.asset_activate operator, then synchronously call sculpt.brush_stroke with float screen/3D samples and restore the exact prior asset reference and active tool. Unsupported asset or View3D contexts report an actionable warning and make no change. Dyntopo is an explicit safety boundary: the native stroke is not started while Dyntopo is active, because the long-lived Guided Ridge modal cannot safely own a nested BMLog transaction; this leaves the mesh and Undo stack unchanged. RUNNING_MODAL/PASS_THROUGH are rejected as incomplete. Smart Fill owns an explicit modal session; mode/object/area/region changes synchronously cancel its visible state while retaining the live handler owner until the next terminal modal event. Reload/install is not considered safe until two separate Blender main-loop grace ticks have elapsed after terminal return. A normal add-on reload/disable performs complete registered-class/keymap/handler teardown. If Blender gives no subsequent modal event during an unload boundary, the exact orphan owner may be retired as idempotent bookkeeping only; this is not a native terminal return, and manual disable while a modal is active remains unsupported. A failed brush/tool restoration remains pending for retry instead of being discarded. Restoration retries only in the exact saved window/workspace/area/region/scene/object context; a mismatched workspace or context is left untouched and remains pending. The reported native mode-switch crash boundary is excluded from automated GUI tests; native verification remains pending.
- File load, Undo-driven target changes, mode/object/mesh/visibility/topology changes, and add-on disable clean up temporary state, predictions, draw handlers, and isolation.
- Esc, right click, mode/object/mesh changes, and add-on unload restore the user's brush asset reference and active tool. Restoration retries only against the exact saved window/workspace/area/region/scene/object context; a mismatched workspace is left untouched and remains pending. No user or custom MFO brush/paint-curve datablock is created or removed. Before apply the mesh remains unchanged; after apply Blender's standard Undo is authoritative. Blender 5.2 Sculpt brush pointer is read-only, so activation uses the supported public asset operator rather than pointer assignment.
