[TOC]



# Repo

- https://github.com/madmaze/pytesseract
- https://github.com/roatienza/straug
- https://github.com/open-mmlab/mmdetection
- https://github.com/JaidedAI/EasyOCR
- PaddleOCR 文字检测，文字识别，关键信息抽取，文档版面解析，文档表格解析 https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/open-mmlab/mmocr
- layout-parsr https://github.com/Layout-Parser/layout-parser
- https://github.com/tesseract-ocr/tesseract
  - GUI工具 https://github.com/manisandro/gImageReader

- https://github.com/JiaquanYe/TableMASTER-mmocr
  - 平安产险提出TableMASTER

- [竞赛总结：科大讯飞2023 表格结构识别挑战赛](https://mp.weixin.qq.com/s/tXDmOi-K7So_XWvvZHKkxQ)

# Paper

- [OCR技术发展综述与实践](https://mp.weixin.qq.com/s/Wf6zmy1PNwnrG8G_RMH4qQ)

- OCR-free Document Understanding Transformer
  - https://github.com/clovaai/donut

- End-to-end object detection with Transformers
  - https://github.com/facebookresearch/detr

- An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition
  - CRNN
  - 先使用cnn进行特征抽取，得到高度为1的特征图，然后输入到双向lstm中，最后经过一个ctc模块计算loss
  - ctc loss的计算：关键是计算所有有效的对齐路径的得分总和，一般使用动态规划的方法计算loss
  - [CTC算法详解](https://zhuanlan.zhihu.com/p/88645033)

# Datasets

- [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)
- docbank 文本和layout分析文档数据集  https://github.com/doc-analysis/DocBank
- 无标注的CDIP文档数据集 https://ir.nist.gov/cdip/
- Layout检测，中文数据集 https://github.com/buptlihang/CDLA 
- macrosoft文档智能开源数据 https://github.com/orgs/doc-analysis/repositories
- 文档阅读顺序数据集 https://github.com/doc-analysis/ReadingBank
- http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
