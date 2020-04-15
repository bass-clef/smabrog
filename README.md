
smabrog
===

## Description
大乱闘スマッシュブラザーズSpecial (super smash brothers special) の為の自動戦績保存/送信ツール

[![スマブラSPの戦歴を自動で保存してくれるツール](https://img.youtube.com/vi/sIGOL1XGylY/0.jpg)](https://www.youtube.com/watch?v=sIGOL1XGylY)

## Usage
- 忙しい人向け
    - 最新の smabrog.zip を [ここから](https://drive.google.com/drive/folders/1-IiiCSpREpDFTm-W0emJHehfb0Fduit4?usp=sharing) DL,解凍して
    - スイッチ、スマブラ、キャプチャソフトをつけて smabrog/smabrog.exe を実行

- 導入の方法

    0. 必要なもの
        - Switchおよび大乱闘スマッシュブラザーズSpecial
        - PCおよび任意のキャプチャソフト
        - smabrog.exe
    1. キャプチャソフト と スマブラSP を起動して オンラインの [READY to FIGHT]が表示された画面にします
        - マスターハンドが乗っかっていない赤のグラデーションの [READY to FIGHT] のほうが検出率が高いです
    2. [smabrog.zip](https://drive.google.com/drive/folders/1-IiiCSpREpDFTm-W0emJHehfb0Fduit4?usp=sharing) を解凍して smabrog/smabrog.exe を起動すると、初回起動のみ、解像度を [16:9] をベースにして FHD(1920x1080) まで自動で再比較、検出します
        - もしも解像度を固定にしたい場合は **smabro.exeが終了後に** config.json の 'every_time_find_capture_area' を false に設定してください
        - [16:9] 以外の解像度は検出率が低くなるかもしれないですけど、config.json の capture.width-height を任意の値にすると、もしかしたら検出されるかもしれないです。
    3. 自動でキャプチャ画面を捕捉します (誤検出されないように他のウィンドウを最小化または閉じておく事をおすすめします)
    4. READY to FIGHT!
- 終了するには

表示される GUI の[x]ボタン か console を ```ctrl+C``` で 閉じてください。

- 戦績を参照する
    - ./log/ フォルダに {日付}.json のファイルが転がっているのでそれを開く
    - 下記のオプションの欄を参考にして **閲 覧 サ イ ト を 作 成 し て も ら い** 後にその URL に対して json を送りつけて見に行く！

- オプション
    - config.json の記述
    ```json
        capture                 /* キャプチャに関する */
            title               /* キャプチャソフトのタイトル */
            x,y,width,height    /* デスクトップの絶対座標 */
        log
            data
                /* *.log の名前を変更する場合に使用。使用できるプレースホルダは下記の通りです。
                 * {now}            # 試合終了時刻 YYYYMMDDHHMMSS 空白は 0 埋めします
                 * {chara[{index}]} # キャラクター配列 [0-3]
                 */
        resource                /* キャラクター一覧 */
        option
            find_capture_area             /* width,height を用いて自動でキャプチャエリアを捕捉する */
            exit_not_found_capture_title    /* title が指定されている状態でそのキャプチャソフトがない場合終了するかどうか */
            battle_informationCUI        /* 試合中の情報を CUI で表示するかどうか　*/
            battle_informationGUI        /* 試合中の情報を GUI で表示するかどうか */
            battle_information
                cui_text        /* CUI に表示する内容 */
                                /* プレースホルダについてはソースの「表示する情報」を参照 */
                gui_text        /* GUI に表示する内容 */
                gui             /* GUI の大きさ */
                pos             /* 情報を表示する座標 */
                caption         /* GUIのタイトル */
                color           /* 文字色 */
                back            /* 背景色 */
                tickness        /* 文字の太さ */

    ```
    - log.json の記述
    ```json
        now                     /* 試合終了時刻 */
        player                  /* プレイヤー配列 [0-3] */
            0
                name            /* キャラクター */
                order           /* 順位 (team戦の場合は 同順が入ってくる事に注意) */
                power           /* 戦闘力 */
                stock           /* 最終的な残機 */
                group           /* チーム戦の場合のチームカラー */
            .. {max 1 or 3}
        rule
            name                /* ルール名 [stock|time] */
            group               /* グループ名 [smash|team] */
            stock               /* 最大数(stockの場合) */
            time                /* "MM:SS.NN" の形式で開始と終了の時間配列 [0-1] {未検出="00:00.00"} */
    ```

### Q&A
- Q. スマブラが検出されない
    - A. 起動してから初期化中に **not found capture area.** と表示されている場合キャプチャソフトの画面を捕捉できていない可能性があります。
        => Q. 検出率を上げるには

- Q. 検出率を上げるには
    - A. 下記の対処法が考えられます
        - キャプチャソフトの解像度を[16:9]にすること
        - [READY to FIGHT]の画面がはっきり表示されていること
        - キャプチャソフト や smabrog.exe 以外のソフトを起動していないこと
- Q. 試合結果がうまく検出されない
    - A. FS_READY_TO_FIGHT の検出率が 0.99 以上なのを確認して **自分の順位や戦闘力が表示されてる画面** をいつもよりゆっくり進んでいくとより検出できるようになります
        - または[連戦を続けますか or READY to FIGHT]の画面で誤検出した時用の処理をしているのでその画面で保存されなければ、全くに近いほど検出されなかった事を示しているので、原因が知りたい場合は作者にlogを提出してみてください

## Author/Licence
- [Humi@bass_clef_](https://twitter.com/bass_clef_)
- [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract#license)
- [大乱闘スマッシュブラザーズ SPECIAL](https://www.smashbros.com/ja_JP/)  
    smabrog に使用しているゲーム内画像の著作権、商標権その他の知的財産権は、当該コンテンツの提供元に帰属します

## Special Thanks
- カービィを使って youtube に動画を上げてくれた方々、デバッグに大変お世話になりました！

## log
    ver = major.minor.build
    major : 下位互換性がない変更 (例:保存しているjsonの記述に読み取れなくなる変更があったなど)
    minor : majorの中で一応安定しているバージョン
    build : pyinstallerするたびに加算されていく数値
- 2020/3/5  
    first commit
- 2020/3/23  
    直接キャプチャしていない環境(youtubeの編集済みのものなど)を使用して判断した場合で,正常にキャラクター名が取得できない可能性があるので検出率の 10% を(他のキャラクターとして検出)するより,50%にあげて誤検出した場合を保存するように変更
- 2020/3/27  
    プレイヤー人数を4人に対応
- 2020/3/28  
    ver 0.5 release

- 2020/3/29  
    結果画面の取得を 順位が両方取れて戦闘力が片方でも取れると検出するように変更
- 2020/3/30  
    GUI_text を PIl.draw にして日本語に対応しました
- 2020/4/2  
    - fix:
        - 検出位置が軽微にずれていた
        - 試合開始の検出率をあげる
        - 戦歴の試合数の数が一致してなかった
    - 一部の戦歴をアニメーションで表示するようにしました
- 2020/4/3  
    ver 0.6 release
- 2020/4/6  
    「再戦しますか」のフレームが検出される値を97%から98%にしました
- 2020/4/7  
    - ストックと制限時間を試合開始のフレームで検出するようにしました
    - 画像の保存を別プロセスでするようにしました
- 2020/4/8  
    グラフが正常な値で描画されるようにしました
- 2020/4/9  
    戦歴をキャラクター別で取得するようにしました
- 2020/4/10  
    - ver 0.7 release
    - logが全くなかった時にも起動できるようにしました
- 2020/4/11  
    イギーを追加しました
- 2020/4/14  
    キャプチャエリアの検出をマルチスレッド処理に変更したのに伴い、下記の変更をしました
    - 未検出時に次回起動時に再検出するように config を変更していたのを削除
    - every_time_find_capture_area が true の場合起動時にも解像度を初期化
    - config に max_workers のオプションを追加 : マルチスレッドのスレッド上限の値
- 2020/4/15  
    - 結果画面の検出をマルチプロセスで処理するようにしました
    - [ready ok]画面の検出率を向上しました
