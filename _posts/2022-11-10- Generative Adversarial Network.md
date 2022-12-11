---
title:  "Generative Adversarial Network"
excerpt: "Deep Learning Summary"

categories:
  - A.I

tags:
  - [deep learning, machine learning, AI, School Summary]

toc: true
toc_sticky: true
layout: post
use_math: true
 
date: 2022-11-10
last_modified_at: 2022-11-10
---

# **GAN<sup>Generative Adversarial Network</sup>**
### 적대적 생성 신경망

### GAN의 원리

### GAN의 이익함수
- 판별자와 생성자 각각의 아웃풋에 대한 값에 대해 
- 판별자의 이익함수
  >$$

# 기본 GAN의 문제점 시험에 나옴


# LSGAN/최소제곱 경쟁적 생경망
- 뭔소리냥...
- 모드 붕괴 현상이 발생하여 하나의 값만 생성
- 이거 사진이 모드 붕괴현상이 발생해 이상한걸 만들었음에도 불구하고 판별자가 너무못해서 제대로 학습안됨, 그래도 기본GAN보다 더 잘함


# DCGAN
- CNN을 GAN에 적용하여 고화질 이미지를 생성할 수 있는 모형
- 기존 GAN들보다 복잡한 이미지에 대해 안정된 결과를 보장

### 특징

- 판별자의 풀링층을 보폭이 2보다 큰 보폭을 사용하는 보폭 합성곱 층으로 대체
- 생성자의 풀링층을 보폭이 1보다 작은 분수 보폭을 적용하는 전치 합성곱층으로 대체

### 전치 합성곱 연산
- 컨볼루션 연산은 선형연산
- 중간중간 비선형연산(ReLU등)을 사용하는 이유는 선형 연산으로만 신경망을 쌓으면 신경망이 하나로 함축될 수 있어 여러개의 층으로 쌓는 이유가 없음. 이때 이것을 신경망이 무너진다?라고 함
- 합성곱 연산은 행렬의 곱으로 나타낼 수 있다
  - 이거 하는법 대충 이해하고 담에 정리

### DCGAN 벡터연산
의미론적으로 생성자의 입력인 노이즈 벡터끼리 계산이 가능하다.
- 물론 잘 안됨


# 와서스타인 경쟁적 생성망

### 와서스타인 정보량

- 이게 무슨뜻일까...?
- 판별자를 비판자라고 부른다


# C<sup>Condition</sup>GAN
- 데이터 생성시 원하는 이미지 생성 할 수 있는 조절 기능 추가
- 기본 GAN에서 label정보가 들어간 GAN
- 판별자를 학습시킬대 특정 라벨의 데이터만 추출하여 학습
  - 그럼 이때 가짜 정보를 라벨에 따라 한 루틴마다 모두 학습하나요?
- 생성자의 입력 잡음분포생성 수 뒤에 라벨 정보를 붙여서 생성자의 입력으로 사용한다.
  - 라벨 정보는 숫자이고 잡음분포는 실수인데 이거 그대로 붙이나?

# InfoGAN
- 데이터 생성시 원하는 이미지 생성
- 판별자를 건들진 않음
- 생성자 입력에 잡음벡터와 잠제코드 c의 결합에 기초하여 가짜 데이터 생성
- 잠재코드 c가 데이터에 존재하는 특징을 가질 수 있게 학습한다.