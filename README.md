
smabrog
===

## Description
大乱闘スマッシュブラザーズSpecial (super smash brothers special) の為の戦績保存ツール

## Usage
- 導入の方法

    0. 必要なもの
        - Switchおよび大乱闘スマッシュブラザーズSpecial
        - PCおよび任意のキャプチャソフト
        - smabrog.exe
    1. キャプチャソフト と スマブラSP を起動して オンラインの [READY to FIGHT]が表示された画面にします
    2. [smabrog.zip](https://drive.google.com/file/d/1e28-5FS7v3A3uT0s7DQTF_Y8wr7MguDX/view?usp=sharing) を解凍して smabrog.exe を起動すると、初回起動のみ、解像度を [16:9] をベースにして FHD(1920x1080) まで自動で再比較、検出します
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
            time                /* {[7|5|3]minute 未実装} */
    ```

## Author/Licence
- [Humi@bass_clef_](https://twitter.com/bass_clef_)
- [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)

- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract#license)

## log
- 2020/3/5  first commit
- 2020/3/23
    直接キャプチャしていない環境(youtubeの編集済みのものなど)を使用して判断した場合で,正常にキャラクター名が取得できない可能性があるので検出率の 10% を(他のキャラクターとして検出)するより,50%にあげて誤検出した場合を保存するように変更
- 2020/3/27
    プレイヤー人数を4人に対応
