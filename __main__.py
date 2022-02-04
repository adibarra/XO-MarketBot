# XO-MarketBot - __main__.py
# @author github/adibarra

from pynput import keyboard
import numpy as np
import pytesseract
import pyautogui
import traceback
import random
import time
import math
import cv2
import sys
import os


# move the mouse more naturally, less like a bot
def moveMouse(x2, y2, img_path=None):
    def point_dist(x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2+(y2-y1)**2)

    # create evenly spaced inner points
    x1, y1 = pyautogui.position()
    ctrlpoints = int(max(point_dist(x1,y1,x2,y2)/50,2))
    x = np.linspace(x1, x2, num=ctrlpoints, dtype='int')
    y = np.linspace(y1, y2, num=ctrlpoints, dtype='int')

    # randomize inner points
    rnd = 5
    xr = [random.randint(-rnd, rnd) for k in range(ctrlpoints)]
    yr = [random.randint(-rnd, rnd) for k in range(ctrlpoints)]
    xr[0] = yr[0] = xr[-1] = yr[-1] = 0
    x += xr
    y += yr

    if img_path != None:
        out_img = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        drawStatus(out_img, status='moving mouse to ('+str(x2)+','+str(y2)+')')
        cv2.imwrite(img_path, out_img)

    # move mouse between points
    points = list(zip(x,y))
    duration = 0.3/len(points)
    for point in points:
        pyautogui.moveTo(*point, duration=duration+random.uniform(0.01,0.05))


# click the mouse more naturally, less like a bot
def clickMouse(rightClick=False):
    if rightClick:
        pyautogui.rightClick(duration=random.uniform(0.05,0.25))
    else:
        pyautogui.leftClick(duration=random.uniform(0.05,0.25))


# type hotkeys more naturally, less like a bot
def pressKeyCombo(*keys):
    pyautogui.hotkey(*keys,interval=random.uniform(0.1,0.25))


# type text more naturally, less like a bot
def typeString(toWrite):
    for char in toWrite:
        pyautogui.write(char, interval=random.uniform(0.01,0.09))


# draw a point
def drawPoint(img_path, x, y):
    img_bgr = cv2.imread(img_path)
    cv2.rectangle(img_bgr, (x, y), (x, y), (0,255,0), thickness=5)
    cv2.imwrite(img_path, img_bgr)


# draw a point
def drawRect(img_path, x1, y1, x2, y2):
    img_bgr = cv2.imread(img_path)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255,0,0), thickness=1)
    cv2.imwrite(img_path, img_bgr)


# wait specified time
def wait(out_path, waitTime, noUpdate=False):
    if not noUpdate:
        img = cv2.imread(out_path)
    counter = 0
    while counter < waitTime:
        remaining = round(waitTime-counter,1)
        if remaining <= 1:
            counter += 0.1
            time.sleep(0.1)
        elif remaining <= 3:
            if not noUpdate:
                drawStatus(img, status='Waiting for '+str(remaining)+' seconds')
                cv2.imwrite(out_path, img)
            counter += 0.5
            time.sleep(0.5)
        else:
            if not noUpdate:
                drawStatus(img, status='Waiting for '+str(remaining)+' seconds')
                cv2.imwrite(out_path, img)
            counter += 1
            time.sleep(1)


# draw an image in the bottom left corner along with its path and a status message but does not write it to file
def drawStatus(img_bgr, subimg_path=None, status='no status message'):
    h, w = img_bgr.shape[:2]
    if subimg_path != None:
        subimg_bgr = cv2.imread(subimg_path)
        sh, sw = subimg_bgr.shape[:2]
        borderThickness = 3
        img_bgr = cv2.rectangle(img_bgr, (0,h-60), (w, h), (0,0,0), thickness=-1)
        img_bgr = cv2.rectangle(img_bgr, (0, h-sh-borderThickness), (sw+borderThickness, h), (255,0,255), thickness=borderThickness)
        img_bgr = cv2.putText(img_bgr, subimg_path, (sw+10,h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0,0,0), thickness=4, lineType=cv2.LINE_AA)
        img_bgr = cv2.putText(img_bgr, status, (sw+10,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0,0,0), thickness=4, lineType=cv2.LINE_AA)
        img_bgr = cv2.putText(img_bgr, subimg_path, (sw+10,h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(150,0,150), thickness=2, lineType=cv2.LINE_AA)
        img_bgr = cv2.putText(img_bgr, status, (sw+10,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(255,0,255), thickness=2, lineType=cv2.LINE_AA)
        img_bgr[h-sh-borderThickness+1:h-borderThickness+1,0+borderThickness:sw+borderThickness,:] = subimg_bgr[:,:,:]
    else:
        img_bgr = cv2.rectangle(img_bgr, (0,h-60), (w, h), (0,0,0), thickness=-1)
        img_bgr = cv2.putText(img_bgr, status, (0+10,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0,0,0), thickness=4, lineType=cv2.LINE_AA)
        img_bgr = cv2.putText(img_bgr, status, (0+10,h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(255,0,255), thickness=2, lineType=cv2.LINE_AA)


# get a point within a bouding box, optionally offset by a multiplied vector
def getRandPointIn(out_path, xmin, ymin, xmax, ymax, offsets=(1.0,1.0,1.0,1.0)):
    xmina = int(offsets[0]*(xmin))
    ymina = int(offsets[1]*(ymin))
    xmaxa = int(offsets[2]*(xmax))
    ymaxa = int(offsets[3]*(ymax))

    # add extra padding to click area just in case
    safetyPadding = 2
    x = int(random.uniform(xmina+safetyPadding,xmaxa-safetyPadding))
    y = int(random.uniform(ymina+safetyPadding,ymaxa-safetyPadding))
    drawRect(out_path,xmina,ymina,xmaxa,ymaxa)
    return x,y


# deprecated - find an image within a bigger image
def findSubimagelegacy(img, subimg_path, out_path, matchNum=0, silent=False, noFailDraw=False,):
    try:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        subimg = cv2.imread(subimg_path,0)
        w, h = subimg.shape[::-1]

        result = cv2.matchTemplate(img_gray,subimg,cv2.TM_CCOEFF_NORMED)

        threshold = 0.80
        loc = np.where(result >= threshold)

        matches = []
        for pt in zip(*loc[::-1]):
            matches.append(pt)

        if matchNum == -1:
            match = matches[0]
            matchReturn =[]
            for mtch in matches:
                matchReturn.append((int(mtch[0]), int(mtch[1]), int(mtch[0] + w), int(mtch[1] + h)))
                cv2.rectangle(img, mtch, (mtch[0] + w, mtch[1] + h), (0,0,255), thickness=3)
                cv2.rectangle(img, (int(mtch[0] + w/2), int(mtch[1] + h/2)), (int(mtch[0] + w/2), int(mtch[1] + h/2)), (0,0,255), thickness=5)
            cv2.imwrite(out_path, img)
            return matchReturn
        else:
            match = matches[matchNum]
            cv2.rectangle(img, match, (match[0] + w, match[1] + h), (0,0,255), thickness=3)
            cv2.rectangle(img, (int(match[0] + w/2), int(match[1] + h/2)), (int(match[0] + w/2), int(match[1] + h/2)), (0,0,255), thickness=5)
            cv2.imwrite(out_path, img)
            return int(match[0]), int(match[1]), int(match[0] + w), int(match[1] + h)
    except Exception:
        if not noFailDraw:
            cv2.rectangle(img, (0,0), (5,5), (0,0,255), thickness=3)
            cv2.rectangle(img, (0,0), (0,0), (0,0,255), thickness=5)
            cv2.imwrite(out_path, img)
        if not silent:
           print('Failed to find: '+subimg_path)
        return None


# find an image within a bigger image, regardless of scaling/size difference between the two
def findSubimage(img, subimg_path, out_path, matchNum=0, singleMatch=True, silent=False, selectionConfidence=0.0, drawConfidenceThreshold=0.5, withConfidence=False):
    try:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        subimg = cv2.imread(subimg_path,0)
        subH, subW = subimg.shape[:2]
        found = None

        drawStatus(img, subimg_path=subimg_path, status='locating template image')
        cv2.imwrite(out_path, img)

        # attempt to find matches for subimage after rescaling big image at different scales
        matches = []
        for scale in np.linspace(0.2, 1.6, 20)[::-1]:
            resized = cv2.resize(img_gray, (int(img_gray.shape[1]*scale),int(img_gray.shape[0]*scale)))
            r = img_gray.shape[1] / float(resized.shape[1])

            if resized.shape[0] < subH or resized.shape[1] < subW:
                break

            result = cv2.matchTemplate(resized, subimg, cv2.TM_CCOEFF_NORMED)

            if singleMatch:
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
                matches.append((int(maxLoc[0] * r), int(maxLoc[1] * r), int((maxLoc[0] + subW) * r), int((maxLoc[1] + subH) * r), float(maxVal)))
            else:
                threshold = 0.80
                loc = np.where(result >= threshold)
                for pt in zip(*loc[::-1]):
                    matches.append((int(pt[0] * r), int(pt[1] * r), int((pt[0] + subW) * r), int((pt[1] + subH) * r), float(result[pt[1]][pt[0]]), float(r)))

        # prepare, orgainize, and sort output coord data
        matchReturn = []
        if singleMatch:
            matches = sorted(matches, key=lambda x: x[-1])
        else:
            matches = sorted(matches, key=lambda x: x[-2])

        for mtch in matches:
            if mtch[4] >= selectionConfidence:
                matchReturn.append((*mtch,))
                if mtch[4] >= drawConfidenceThreshold:
                    r = 255 if mtch[4] <= 0.75 else 255*mtch[4]
                    g = 255*mtch[4] if mtch[4] <= 0.75 else 255
                    b = 0
                    cv2.rectangle(img, (mtch[0], mtch[1]), (mtch[2], mtch[3]), (b,g,r), thickness=3)
                    cv2.rectangle(img, (mtch[2], mtch[3]), (mtch[2], mtch[3]), (b,g,r), thickness=5)
                    img = cv2.putText(img, str(round(mtch[4]*100,2))+'%', (mtch[2]+10,mtch[3]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(b,g,r), thickness=1, lineType=cv2.LINE_AA)
        drawStatus(img, subimg_path=subimg_path, status='locating template image')
        cv2.imwrite(out_path, img)

        if not silent and len(matchReturn) == 0:
           print('Failed to find: '+subimg_path+' with confidence>='+str(selectionConfidence))
           return None

        if singleMatch:
            if withConfidence:
                return matchReturn
            else:
                return list(ele[:-1] for ele in matchReturn)
        else:
            def calc_best_scale(a, groupColumn=0, avgRow=0):
                scales = []
                data = []
                for row in a:
                    if row[groupColumn] not in scales:
                        scales.append(row[groupColumn])
                        data.append([row[groupColumn],row[avgRow]])
                    else:
                        data.append([row[groupColumn],row[avgRow]])
                averages = [[0,0]]*len(scales)
                for i in range(0,len(scales)):
                    counter = 0
                    for dta in data:
                        if dta[0] == scales[i]:
                            averages[i] = [scales[i],averages[i][1]+dta[1]]
                            counter += 1
                    averages[i][1] = averages[i][1]/counter
                return sorted(averages, key=lambda x: x[-1], reverse=True)
            
            def fuseClosePoints(points, d):
                # https://stackoverflow.com/questions/19375675/simple-way-of-fusing-a-few-close-points
                def dist2(p1, p2):
                    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                ret = []
                d2 = d * d
                n = len(points)
                taken = [False] * n
                for i in range(n):
                    if not taken[i]:
                        count = 1
                        point = list(points[i])
                        taken[i] = True
                        for j in range(i+1, n):
                            if dist2(points[i], points[j]) < d2:
                                point[0] += points[j][0]
                                point[1] += points[j][1]
                                count+=1
                                taken[j] = True
                        point[0] = int(point[0] / count)
                        point[1] = int(point[1] / count)
                        ret.append(point)
                return ret
            
            matchResultsScale = []
            matchReturn = fuseClosePoints(matchReturn, 4)
            bestScale = calc_best_scale(matchReturn, 5, 4)
            for match in matchReturn:
                if match[5] == bestScale[0][0]:
                    matchResultsScale.append(match)
            
            if withConfidence:
                if matchNum == -1:
                    return list(ele[:-1] for ele in matchResultsScale)
                else:
                    return list(ele[:-1] for ele in matchResultsScale)[matchNum]
            else:
                if matchNum == -1:
                    return list(ele[:-2] for ele in matchResultsScale)
                else:
                    return list(ele[:-1] for ele in matchResultsScale)[matchNum]
    except Exception as ex:
        print(''.join(traceback.TracebackException.from_exception(ex).format()))
        if not silent:
           print('Failed to find: '+subimg_path+' with confidence>='+str(selectionConfidence))
        return None


# move to the location of an image on the another image
def moveToImage(big_img, subimg_path, out_path, offsets=(1.0,1.0,1.0,1.0), matchNum=0, silent=False, move=True, selectionConfidence=0.0, drawConfidenceThreshold=0.5, withConfidence=False):
    coords = findSubimage(big_img, subimg_path, out_path=out_path, matchNum=matchNum, silent=silent, selectionConfidence=selectionConfidence, drawConfidenceThreshold=drawConfidenceThreshold, withConfidence=withConfidence)
    
    confidence = None
    if coords != None and len(coords) > 0:
        coords = coords[-1]
        if withConfidence:
            confidence = coords[-1]
            coords = coords[:-1]
        
        coords = getRandPointIn(out_path, *coords, offsets=offsets)
        drawPoint(out_path,*coords)
        if move:
            moveMouse(*coords)
    return confidence


# translate an image into a string
def tesseract_img2str(ocr_img, out_path, coords, offsets=(1.0,1.0,1.0,1.0), digitsOnly=False, resultToRight=False):
    # preprocess img
    img_orig = cv2.cvtColor(np.array(ocr_img), cv2.COLOR_RGB2BGR)
    if coords == None:
        coords = (0,0,img_orig.shape[1],img_orig.shape[0])
    crop = (tuple(map(lambda i, j: int(i * j), coords, offsets)))
    img = img_orig[crop[1]:crop[3], crop[0]:crop[2]]
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(src=img)
    img = cv2.resize(src=img, dsize=(img.shape[1]*5,img.shape[0]*5))
    img = cv2.threshold(src=img, thresh=190, maxval=255, type=cv2.THRESH_BINARY)[1]

    # highlight where ocr will take place and its result
    out_img = cv2.imread(xocv_img_path)
    cv2.rectangle(out_img, (crop[0],crop[1]), (crop[2],crop[3]), (255,255,255), thickness=2)
    cv2.imwrite(out_path, out_img)

    # initiate ocr
    pytesseract.pytesseract.tesseract_cmd = os.path.dirname(os.path.realpath(sys.argv[0]))+r'\tesseract-ocr\tesseract'
    if digitsOnly:
        output = str(pytesseract.image_to_string(img, lang='eng', config='--psm 7 -c tessedit_char_whitelist="1234567890,.@"'))
        def tesseractCustomNum(strToTest) -> bool:
            if str.isdigit(strToTest) or strToTest in [',','.','@']:
                return True
            return False
        output = ''.join(filter(tesseractCustomNum, output))
    else:
        output =  str(pytesseract.image_to_string(img, lang='eng', config='--psm 7')).replace('\n\x0c','')
    
    # display ocr result
    if resultToRight:
        offset = (-crop[0]+crop[2]+10,-30)
    else:
        offset = (0,0)
    out_img = cv2.putText(out_img, 'OCR: '+repr(output), (crop[0]+offset[0],crop[3]+25+offset[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,0), thickness=5, lineType=cv2.LINE_AA)
    out_img = cv2.putText(out_img, 'OCR: '+repr(output), (crop[0]+offset[0],crop[3]+25+offset[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,140,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path, out_img)
    return output


# having been passed an image of the trade page for an item in xo, look for the item's buy orders
def readBuyOrders(big_img, out_path, silent=False):
    coords = findSubimage(big_img, 'res/landmarks/coin.png', out_path, matchNum=-1, singleMatch=False, silent=silent)
    orderbook = []
    for coord in coords:
        # filter to buy orders section of sceen
        if coord[0] > big_img.size[0]/2 and coord[0] < 17*big_img.size[0]/20 and coord[1] > 12*big_img.size[1]/20 and coord[1] < 15*big_img.size[1]/20:
            quantity = tesseract_img2str(big_img, out_path, coord, offsets=(0.925,1.0,1.275,1.0), digitsOnly=True, resultToRight=True)
            quantity = quantity.replace('\n',' ')
            if quantity.startswith('0') and not quantity.startswith('0.'):
                quantity = '0.'+quantity[1:]
            orderbook.append(quantity)
    return sorted(orderbook, key=lambda x: float(str(x).split('@')[0]))


# having been passed an image of the trade page for an item in xo, look for the item's sell orders
def readSellOrders(big_img, out_path, silent=False):
    coords = findSubimage(big_img, 'res/landmarks/coin.png', out_path, matchNum=-1, singleMatch=False, silent=silent)
    orderbook = []
    for coord in coords:
        # filter to buy orders section of sceen
        if coord[0] > 3*big_img.size[0]/20 and coord[0] < big_img.size[0]/2 and coord[1] > 12*big_img.size[1]/20 and coord[1] < 15*big_img.size[1]/20:
            quantity = tesseract_img2str(big_img, out_path, coord, offsets=(0.8,1.0,1.74,1.0), digitsOnly=True, resultToRight=True)
            quantity = quantity.replace('\n',' ')
            if quantity.startswith('0') and not quantity.startswith('0.'):
                quantity = '0.'+quantity[1:]
            orderbook.append(quantity)
    return sorted(orderbook, key=lambda x: float(str(x).split('@')[0]), reverse=True)


# calculate the optimal price to pay for an item based on current orders and a preset max acceptable price
def calculateTargetPrice(orderbook, max_buy_price, increment=0.01, price_up_amount=20):
    target_buy_price = 0
    largest_buy_price = max_buy_price if len(orderbook) == 0 else float(str(orderbook[-1]).split('@')[0].strip())
    largest_buy_price_amount = 50 if len(orderbook) == 0 else int(str(orderbook[-1]).split('@')[1].strip())
    second_largest_buy_price_amount = largest_buy_price_amount if len(orderbook) <= 1 else int(str(orderbook[-2]).split('@')[1].strip())

    if largest_buy_price > max_buy_price:
        if largest_buy_price_amount < price_up_amount:
            if largest_buy_price - increment <= max_buy_price:
                target_buy_price = largest_buy_price - increment
            else:
                target_buy_price = max_buy_price
        else:
            target_buy_price = max_buy_price
    elif largest_buy_price == max_buy_price:
        if largest_buy_price_amount <= price_up_amount/2 and second_largest_buy_price_amount <= price_up_amount:
            target_buy_price = largest_buy_price - increment
        else:
            target_buy_price = largest_buy_price
    else:
        if largest_buy_price_amount > price_up_amount:
            target_buy_price = largest_buy_price + increment
        elif largest_buy_price_amount <= price_up_amount/2:
            target_buy_price = largest_buy_price - increment
        else:
            target_buy_price = largest_buy_price
    return round(target_buy_price, 2)


# having been passed an image of the myoffers tab for in xo, look for the item's offers and orders
def readMyOffersTab(big_img, subimg_path, out_path, silent=False):
    coords = findSubimage(big_img, subimg_path, out_path, matchNum=-1, singleMatch=False, silent=silent)
    offerbook = []
    orderbook = []
    for coord in coords:
        offers = (
            tesseract_img2str(big_img, out_path, coord, offsets=(3.5,1.015,2.82,1.025), digitsOnly=True).replace('\n',' ')
            +tesseract_img2str(big_img, out_path, coord, offsets=(4.6,1.015,3.0,1.025), digitsOnly=True).replace('\n',' '))
            
        orders = (
            tesseract_img2str(big_img, out_path, coord, offsets=(5.2,1.015,3.87,1.025), digitsOnly=True).replace('\n',' ')
            +tesseract_img2str(big_img, out_path, coord, offsets=(6.3,1.015,4.1,1.025), digitsOnly=True).replace('\n',' '))
        
        if offers.startswith('0') and not offers.startswith('0.'):
            offers = '0.'+offers[1:]
        if orders.startswith('0') and not orders.startswith('0.'):
            orders = '0.'+orders[1:]
        
        if offers != '':
            offerbook.append([offers,coord])
        if orders != '':
            orderbook.append([orders,coord])
    return sorted(offerbook, key=lambda x: float(str(x[0]).split('@')[0])),sorted(orderbook, key=lambda x: float(str(x[0]).split('@')[0]))


# check myoffers tab for offers previously placed which are no longer optimal
def cancelStaleOrders(subimg_path, out_path, item_target_cost, silent=True):
    for i in range(40): #finite loop instead of while loop to avoid issues, max of 20 offers and 20 orders = 40 loops
        offerbook, orderbook = readMyOffersTab(pyautogui.screenshot(), subimg_path, out_path, silent=silent)
        if len(orderbook) == 0:
            break

        found = False
        for order in orderbook:
            if float(order[0].split('@')[0]) < item_target_cost:
                found = True
        if not found:
            break

        for order in orderbook:
            if float(order[0].split('@')[0]) < item_target_cost:
                moveMouse(order[1][0], order[1][1])
                clickMouse(rightClick=True)
                wait(xocv_img_path, 0.5)
                moveToImage(pyautogui.screenshot(), 'res/buttons/cancel_offer.png', out_path, matchNum=-1)
                clickMouse()
                wait(xocv_img_path, 3)
                moveToImage(pyautogui.screenshot(), 'res/buttons/yes.png', out_path, offsets=(1.0,1.0,1.08,1.0), matchNum=-1)
                clickMouse()
                wait(xocv_img_path, 3)
                break



#####                #####
### Script starts here ###
#####                #####
pyautogui.FAILSAFE = False
xocv_img_path = 'xocv.png'
couponsx10_max_cost = 0.23
couponsx10_target_cost = 0.23
tot_quantity = 0



def on_press(key):
    if key == keyboard.Key.f2:
        listener.stop()
        print('exited')
        os._exit(0)

listener = keyboard.Listener(
    on_press=on_press)
listener.start()



wait(xocv_img_path, 3, noUpdate=True)
print('running')

moveToImage(pyautogui.screenshot(), 'res/buttons/close_news.png', xocv_img_path, offsets=(1.0, 1.0, 1.05, 1.0), silent=True, selectionConfidence=0.65)
clickMouse()
moveMouse(random.randint(0,10),random.randint(0,10), xocv_img_path)

moveToImage(pyautogui.screenshot(), 'res/tabs/garage.png', xocv_img_path, matchNum=-1)
clickMouse()
moveMouse(random.randint(0,10),random.randint(0,10), xocv_img_path)

moveToImage(pyautogui.screenshot(), 'res/tabs/market.png', xocv_img_path, matchNum=-1)
clickMouse()
moveMouse(random.randint(0,10),random.randint(0,10), xocv_img_path)
wait(xocv_img_path, 1)

while True:

    # begin locating search bar
    moveToImage(pyautogui.screenshot(), 'res/subtabs/parts.png', xocv_img_path, matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 3)

    tmp_sc = pyautogui.screenshot()
    attempts = [
        'res/buttons/search_blank.png',
        'res/buttons/search_coupons.png',
        'res/buttons/search_legend.png'
    ]
    mostConf = None
    for i in range(0,len(attempts)):
        conf = moveToImage(tmp_sc, attempts[i], xocv_img_path, offsets=(1.0, 1.0, 0.90, 1.0), matchNum=-1, silent=True, move=False, withConfidence=True)
        if conf != None and (mostConf == None or conf > mostConf[1]):
            mostConf = (i, conf)

    # search for item to purchase
    moveToImage(tmp_sc, attempts[mostConf[0]], xocv_img_path, offsets=(1.0, 1.0, 0.90, 1.0), matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 0.5)
    pressKeyCombo('ctrl','a')
    typeString('coupons')
    pressKeyCombo('enter')
    wait(xocv_img_path, 3)

    #moveToImage(pyautogui.screenshot(), 'res/buttons/filter_bar.png', xocv_img_path, offsets=(5.70, 1.0, 0.743, 1.0), matchNum=-1)
    #clickMouse()
    #wait(xocv_img_path, 3)

    moveToImage(pyautogui.screenshot(), 'res/buttons/couponsX10.png', xocv_img_path, offsets=(0.835, 0.98, 1.75, 1.06), matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 3)

    # read in orderbook and calculate optimal order price
    moveToImage(pyautogui.screenshot(), 'res/buttons/reload.png', xocv_img_path, matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 3)

    orderbook = readBuyOrders(pyautogui.screenshot(), xocv_img_path)
    wait(xocv_img_path, 1)
    
    # input order
    moveToImage(pyautogui.screenshot(), 'res/buttons/buy.png', xocv_img_path, offsets=(1.0, 1.0, 1.12, 1.0), matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 2)

    moveToImage(pyautogui.screenshot(), 'res/buttons/cost_input.png', xocv_img_path, offsets=(1.2, 1.0, 0.905, 1.0), matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 0.25)
    pressKeyCombo('ctrl','a')
    wait(xocv_img_path, 0.25)
    current_cost_couponsx10 = calculateTargetPrice(orderbook, couponsx10_max_cost, increment=0.01, price_up_amount=20)
    typeString(str(current_cost_couponsx10))
    wait(xocv_img_path, 0.5)

    moveToImage(pyautogui.screenshot(), 'res/buttons/quantity_input.png', xocv_img_path, offsets=(1.072, 1.043, 0.95, 1.0), matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 0.25)
    pressKeyCombo('ctrl','a')
    wait(xocv_img_path, 0.25)
    typeString('20')
    wait(xocv_img_path, 0.6, noUpdate=True) # wait until cursor blink after typing for more reliable ocr below
    
    tmp_sc = pyautogui.screenshot()
    coords = findSubimage(tmp_sc, 'res/buttons/quantity_input.png', xocv_img_path, matchNum=-1)[-1]
    quantity = tesseract_img2str(tmp_sc, xocv_img_path, coords, offsets=(1.075,1.05,0.81,0.985), digitsOnly=True)
    wait(xocv_img_path, 2)
    
    if quantity == '' or int(quantity) == 0:
        moveToImage(pyautogui.screenshot(), 'res/buttons/close.png', xocv_img_path, offsets=(1.0, 1.0, 1.06, 1.0), matchNum=-1)
        clickMouse()
        wait(xocv_img_path, 1)
    
    else:
        tot_quantity += int(quantity)
        print('total purchased:',tot_quantity)
        moveToImage(pyautogui.screenshot(), 'res/buttons/order.png', xocv_img_path, offsets=(1.0, 1.0, 1.075, 1.0), matchNum=-1)
        clickMouse()
        wait(xocv_img_path, 5)
    
        moveToImage(pyautogui.screenshot(), 'res/buttons/ok.png', xocv_img_path, offsets=(1.0, 1.0, 1.065, 1.0), matchNum=-1)
        clickMouse()
        wait(xocv_img_path, 1)
    
    moveToImage(pyautogui.screenshot(), 'res/buttons/back.png', xocv_img_path, matchNum=-1)
    clickMouse()
    
    # remove stale orders before next run
    moveToImage(pyautogui.screenshot(), 'res/subtabs/offers.png', xocv_img_path, matchNum=-1)
    clickMouse()
    wait(xocv_img_path, 3)

    cancelStaleOrders('res/buttons/couponsX10.png', xocv_img_path, couponsx10_target_cost)
    wait(xocv_img_path, 3)

    # wait ~ 1 minute before looping
    wait(xocv_img_path, 60 + random.randint(0, 15))
