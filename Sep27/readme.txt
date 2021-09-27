optimization
-changes in patch.py
-changes in images set, move into yolov3 and test
-changes in yolo utils loss.py:
        loss = lbox + lobj + lcls
        return loss * bs, lobj, torch.cat((lbox, lobj, lcls, loss)).detach()
