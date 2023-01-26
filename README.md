# pop or classic

## 入力
- 以下の3つの特徴量を持った256音を入力とする。これらは全て実数値として表されているため、入力は256×3=768次元のベクトルとなる。
    - 音階（note）
    - 直前の音の鳴り始めから何秒後に鳴り始めたか（最初の音は0）
    - 何秒鳴らしているか

## MIDIの仕様
- timeは前回のイベントからどれだけ経過したか。（4分音符が480？）
- velocityは音量の指定（0で音を消すことを意味できる）
- tempoは microseconds/beat
    
    500,000は4分音符が0.5秒鳴らされることを意味する

## 使用したデータセット
- <a href=https://github.com/music-x-lab/POP909-Dataset>POP909</a>
    > Ziyu Wang et al., "POP909: A Pop-song Dataset for Music Arrangement Generation", Proceedings of 21st International Conference on Music Information Retrieval, 2020

- <a href=https://magenta.tensorflow.org/datasets/maestro>The MAESTRO Dataset (V3.0.0)</a>
    > Curtis Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.", In International Conference on Learning Representations, 2019.