# Tacotron Korean TTS implementation using Tensorflow2

### Requirements
* Python = 3.7.x
* tensorflow-gpu = 2.x
* jamo

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

2. **`~/Tacotron-Korean-Tensorflow2`에 학습 데이터 준비**

   ```
   Tacotron-Korean-Tensorflow2
     |- kss
         |- 1
         |- 2
         |- 3
         |- 4
         |- transcript.v.1.x.txt
   ```

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * data 폴더에 학습에 필요한 파일들이 생성됩니다

4. **Train**
   ```
   python train1.py
   python train2.py
   ```
     * train1.py - train2.py 순으로 실행합니다
     * 저장한 모델이 있으면 가장 최근의 모델을 불러와서 재학습합니다

5. **Synthesize**
   ```
   python test1.py
   python test2.py
   ```
     * test1.py - test2.py 순으로 실행하면 output 폴더에 wav 파일이 생성됩니다



윈도우에서 Tacotron 한국어 TTS 학습하기
  * https://chldkato.tistory.com/141
  
Tacotron 정리
  * https://chldkato.tistory.com/143
