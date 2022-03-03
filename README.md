# XO-MarketBot
This project is a bot which is meant to allow users automatically purchase, craft, and sell items at an optimal price in a game called Crossout.

Last updated [2021-09-01].

---

![Screenshot](https://github.com/adibarra/XO-MarketBot/blob/main/xocv.png)

## Features:
- Automatically open market tab
- Automatically read in orderbook via OCR and template matching
- Automatically place orders via OCR and template matching

## Version 1.0 Roadmap:
- [x] Add template matching support
- [x] Add OCR support
- [x] Add OCR image preprocessing
- [x] Upgrade template matching search
- [x] Display a status message
- [x] Display the image being searched for
- [x] Automatic item purchasing
- [x] Periodically reset orders to avoid stuck low orders
- [ ] Improve item price calculator by calculating trade volume
- [ ] Automatic item crafting
- [ ] Automatic item selling
- [ ] Better reliability


## Notes:
- Built with Python 3.9.
- Automatically creates a png file called 'xocv.png' which is updated to display what the bot is currently doing

---

## Installation and Setup Instructions
1. **Install Python Dependencies:**
```bash
    ### install python dependencies
    $ sudo apt-get install python3
    $ python3 -m pip install --upgrade pip
    $ python3 -m pip install -r path/to/project/requirements.txt
```

2. **Install tesseract-ocr Binaries:**
```bash
    ### download tesseract-ocr folder to the project direcory
    $ cd path/to/project
```

3. **Start the bot:**
```bash
    ### run the bot script
    $ cd path/to/project
    $ python3 /path/to/project/__main__.py
```

---
And you're done!
