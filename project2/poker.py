from pydoc import cli
import pygame, sys
pygame.init()

# NOTE: The following implementation is a sample and will not run without 
# the necessary model files (e.g., SAV files).
## ---------------- FILL HERE ---------------------------------
from collections import Counter
import pickle
import numpy as np
import pandas as pd

# 1. features.py 내용 전체 (원본과 100% 동일)
STRAIGHT_SETS = [frozenset(range(low, low + 5)) for low in range(1, 10)]
STRAIGHT_SETS.append(frozenset((1, 10, 11, 12, 13)))

ROYAL_RANKS = frozenset((1, 10, 11, 12, 13))


def hand_category(hand):
    """3~5장의 현재 족보 점수 (0 노페어 ~ 9 로열 플러시)."""

    if len(hand) not in (3, 4, 5):
        raise ValueError("카드는 3장, 4장 또는 5장이어야 합니다.")

    ranks = [rank for suit, rank in hand]
    suits = [suit for suit, rank in hand]
    frequencies = sorted(Counter(ranks).values(), reverse=True)

    if len(hand) == 5:
        rank_set = frozenset(ranks)
        straight = len(rank_set) == 5 and rank_set in STRAIGHT_SETS
        flush = len(set(suits)) == 1

        if straight and flush:
            return 9 if rank_set == ROYAL_RANKS else 8
        if 4 in frequencies:
            return 7
        if frequencies == [3, 2]:
            return 6
        if flush:
            return 5
        if straight:
            return 4

    if 4 in frequencies:
        return 7
    if 3 in frequencies:
        return 3
    if frequencies.count(2) == 2:
        return 2
    if 2 in frequencies:
        return 1
    return 0


FEATURE_COLUMNS = [
    "n_cards",           # 카드 장수 (3 또는 4)
    "rank_max_count",    # 같은 숫자의 최대 장수
    "pair_count",        # 페어 개수
    "triple_count",      # 트리플 개수
    "rank_unique",       # 서로 다른 숫자 개수
    "suit_max_count",    # 같은 문양의 최대 장수
    "suit_unique",       # 서로 다른 문양 개수
    "straight_windows",  # 현재 숫자를 모두 포함하는 STRAIGHT_SETS 개수
    "sf_windows",        # 모두 같은 문양이면 straight_windows, 아니면 0
    "royal_possible",    # 같은 문양 + 숫자가 모두 {A,10,J,Q,K}
    "current_score",     # hand_category(hand)
]


def extract_features(hand):
    """(suit, rank) 튜플 3~4개 → FEATURE_COLUMNS 순서의 정수 리스트."""

    if len(hand) not in (3, 4):
        raise ValueError("현재 카드가 3장 또는 4장이어야 합니다.")

    cards = [(int(suit), int(rank)) for suit, rank in hand]

    for suit, rank in cards:
        if not (1 <= suit <= 4 and 1 <= rank <= 13):
            raise ValueError(f"카드 숫자 또는 문양 범위 오류: {(suit, rank)}")
    if len(set(cards)) != len(cards):
        raise ValueError("동일한 카드가 중복되었습니다.")

    ranks = [rank for suit, rank in cards]
    suits = [suit for suit, rank in cards]
    rank_counts = list(Counter(ranks).values())
    suit_counts = list(Counter(suits).values())
    rank_set = frozenset(ranks)
    suited = len(suit_counts) == 1

    # 숫자가 중복되면 스트레이트 불가 → 0
    if len(rank_set) == len(ranks):
        straight_windows = sum(1 for s in STRAIGHT_SETS if rank_set <= s)
    else:
        straight_windows = 0

    return [
        len(cards),
        max(rank_counts),
        rank_counts.count(2),
        rank_counts.count(3),
        len(rank_set),
        max(suit_counts),
        len(suit_counts),
        straight_windows,
        straight_windows if suited else 0,
        int(suited and rank_set <= ROYAL_RANKS),
        hand_category(cards),
    ]


# 2. 모델 로드 및 확률 예측 함수 (가이드 2-3 준수 + 45개 조합 캐싱 최적화)
with open("poker-hand-model.sav", "rb") as f:
    hand_model = pickle.load(f)

_proba_cache = {}


def predict_hand_proba(hand):
    """현재 패(3~4장) → 최종 5장 족보 0~9 확률 (길이 10 numpy 배열, 합 1)."""
    feat = tuple(extract_features(hand))
    if feat not in _proba_cache:
        X = pd.DataFrame([feat], columns=FEATURE_COLUMNS)
        proba = np.zeros(10)
        proba[hand_model.classes_] = hand_model.predict_proba(X)[0]
        _proba_cache[feat] = proba
    return _proba_cache[feat]


def win_tie_proba(hand, opposite_score):
    """(최종 족보 > 상대 점수일 확률, 최종 족보 == 상대 점수일 확률)."""
    proba = predict_hand_proba(hand)
    return proba[opposite_score + 1:].sum(), proba[opposite_score]


# 3. Call / Fold 베팅 의사결정 함수 (조원 구현부)
def predict_call(hand, opposite_score):
    # 상대방이 0점(노페어)이면 최소 무승부 이상이므로 무조건 Call
    if opposite_score == 0:
        return True

    p_win, p_tie = win_tie_proba(hand, opposite_score)

    # 쇼다운 시 돌려받는 기대 상금: 승리 시 +$6, 무승부 시 +$3
    expected_return = 6.0 * p_win + 3.0 * p_tie

    if len(hand) == 3:
        # 3장 단계: 끝까지 갈 때 추가 비용 $2 대비 4장 확인 후 중도 포기(Fold) 옵션 가치를 반영한 임계값 ($1.8)
        # (순수익 기준 ev_showdown >= -1.2 와 수학적으로 완전히 동일한 식)
        return expected_return >= 1.8
    else:
        # 4장 단계: 마지막 1장 추가 베팅 비용 $1 이상이면 Call
        # (가이드 2-5의 6*p_win + 3*p_tie - 1 >= 0 및 ev_showdown >= -2.0 과 동일)
        return expected_return >= 1.0
## -------------------------------------------------

import pandas as pd
cand_hands = pd.read_csv('poker-hand-testing.csv')

def pick_opposite_score():
    return cand_hands['Poker Hand'].sample().values[0]

def pick_five_cards():
    sam = cand_hands.sample().values[0]
    score = sam[-1]
    lis = sam[:-1]
    return (
        [ (a, b) for a, b in zip(lis[::2], lis[1::2]) ],
        score)


# game money: initially, $1000
money = 1000

# game count
#game_count = 1000

# initially, pick five cards
full_hand, hand_score = pick_five_cards()
print(full_hand, hand_score)

# currently, three flipped cards
flipped = 3
current_cards = full_hand[:flipped]

# current game's opposite hand score
opposite_hand_score = pick_opposite_score()

#game_count = game_count - 1
money = money - 1

font = pygame.font.SysFont("comicsans",20)
SCREEN_WIDTH, SCREEN_HEIGHT = 512, 512

# create window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("POKER")

CARD_WIDTH, CARD_HEIGHT = 60, 100
SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT = 32, 64

# create a surface object, image is drawn on it.
imp = pygame.image.load("cards.jpg")
imp = pygame.transform.scale(imp, (SUIT_ICON_WIDTH*4, SUIT_ICON_HEIGHT))

# list of suit icons (0: hear, 1: spade, 2: diamond, 3: club)
suits = list()

# 0: heart
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*3, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 1: spade
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (0, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 2: diamond
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 3: club
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*2, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (128, 128, 128)

def draw_card(suit, rank, pos):
    # white card
    rect = pygame.Rect( pos, (CARD_WIDTH, CARD_HEIGHT) )
    pygame.draw.rect(screen, white, rect)

    # suit icon
    screen.blit(
        suits[suit], 
        (pos[0] + CARD_WIDTH/2 - SUIT_ICON_WIDTH/2, 
            pos[1] + CARD_HEIGHT*2/3 - SUIT_ICON_HEIGHT/2))

    # rank
    rank_text = font.render("{}".format(rank), True, black)
    rank_rect = rank_text.get_rect()
    rank_rect.center = (pos[0] + CARD_WIDTH/2, pos[1] + CARD_HEIGHT/4)
    screen.blit(rank_text, rank_rect)

def gamemoney_text():
    text = font.render("Your money: ${}".format(money), True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6)
    screen.blit(text, rank_rect)

def opposite_hand_text():
    text = font.render(
        "The opposite's hand score: {}".format(opposite_hand_score), 
        True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6 + rank_rect.height)
    screen.blit(text, rank_rect)

BUTTON_WIDTH, BUTTON_HEIGHT = 120, 50

def draw_button(center_pos, button_text, button_color):
    # rect button
    #rect = pygame.Rect(
    #    (center_pos[0] - BUTTON_WIDTH/2, center_pos[1] - BUTTON_HEIGHT/2),
    #    (BUTTON_WIDTH, BUTTON_HEIGHT) )
    rect = pygame.Rect((0, 0, BUTTON_WIDTH, BUTTON_HEIGHT))
    rect.center = center_pos
    pygame.draw.rect(screen, button_color, rect) 

    # button text
    text = font.render(button_text, True, black)
    text_rect = text.get_rect()
    text_rect.center = rect.center
    screen.blit(text, text_rect)

msg = None
MESSAGE_HEIGHT = 40
def display_message(_msg):
    if _msg is not None:
        # white message board at the top of screen
        rect = pygame.Rect( (0, 0), (SCREEN_WIDTH, MESSAGE_HEIGHT) )
        pygame.draw.rect(screen, white, rect)
        
        # display message text
        text = font.render(_msg, True, black)
        rect = text.get_rect()
        rect.center = (SCREEN_WIDTH/2, MESSAGE_HEIGHT/2)
        screen.blit(text, rect)

# bet and die buttons at the bottom of screen
BET_BUTTON_CENTER = (SCREEN_WIDTH/4, SCREEN_HEIGHT*4/5)
bet_button_color = white

def draw_bet_button():
    draw_button(BET_BUTTON_CENTER, "CALL", bet_button_color)

# answer if pos is on the button
def on_bet_button(pos):
    return (abs(pos[0] - BET_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - BET_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

DIE_BUTTON_CENTER = (SCREEN_WIDTH/2, SCREEN_HEIGHT*4/5)
die_button_color = white

def draw_die_button():
    draw_button(DIE_BUTTON_CENTER, "FOLD", die_button_color)

def on_die_button(pos):
    return (abs(pos[0] - DIE_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - DIE_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# auto play with classifier
CLS_BUTTON_CENTER = (SCREEN_WIDTH*3/4, SCREEN_HEIGHT*4/5)
cls_button_color = white

def draw_cls_button():
    draw_button(CLS_BUTTON_CENTER, "AUTO", cls_button_color)

def on_cls_button(pos):
    return (abs(pos[0] - CLS_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - CLS_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# play sound when cards are revealed
flip_sound = pygame.mixer.Sound("flipcard.wav")
def play_cardflip_sound(rep):
    pygame.mixer.Sound.play(flip_sound, loops=rep-1)

def draw_current_cards():
    for i, (suit, rank) in enumerate(current_cards):
        draw_card(
            suit-1, rank, # suit and rank 
            ((i+1)*SCREEN_WIDTH/7, SCREEN_HEIGHT*2/5))

# game state for updating screen
# 0: beginning of game - three cards are flipped
# 1: bet - one more card is flipped
# 2: bet & game over - the last card is flipped and
#    win/tie/lose messages are shown
#    then, go to the next game after 5 secs
# 3: die & game over - message shown then,
#    go to the next game after 5 secs
# 9: wait for the next action
update_state = 0

def draw_items_on_screen():
    screen.fill(black)
    opposite_hand_text()
    gamemoney_text()
    draw_bet_button()
    draw_die_button()
    draw_cls_button()
    draw_current_cards()

    if msg is not None:
        display_message(msg)

    pygame.display.flip()
    pygame.time.delay(200)

# drawing cards and playing sound
def update_cards_and_sounds():
    global flipped, update_state, full_hand, hand_score,\
        money, msg, current_cards, opposite_hand_score

    if update_state == 0:
        # at the beginning of game, three cards are flipped
        play_cardflip_sound(3)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 1:
        # check button clicked and now four cards revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 2 or update_state == 3:
        # the last cards are revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # delay another 1 sec for showing message
        pygame.time.delay(1000)
        msg = None
        draw_items_on_screen()

        # a new poker hand is drawn
        full_hand, hand_score = pick_five_cards()
        print(full_hand, hand_score)

        # three flipped cards
        flipped = 3
        current_cards = full_hand[:flipped]

        # opposite's hand score
        opposite_hand_score = pick_opposite_score()
        money = money - 1
        update_state = 0


game_stop = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_stop:
            click_pos = pygame.mouse.get_pos()

            auto_bet = False
            auto_die = False
            if on_cls_button(click_pos):
                if predict_call(current_cards, opposite_hand_score):
                    print("auto call!")
                    auto_bet = True
                    display_message("AI calls")
                    pygame.display.update()
                    pygame.time.delay(800)
                else:
                    print("auto fold!")
                    auto_die = True
                    display_message("AI folds")
                    pygame.display.update()
                    pygame.time.delay(800)

            if on_bet_button(click_pos) or auto_bet:
                flipped = flipped + 1
                current_cards = full_hand[:flipped]
                money = money - 1

                # if game over,
                if flipped == 5:
                    # case 1: win
                    if hand_score > opposite_hand_score:
                        money = money + 6
                        msg = "Win"
                        update_state = 2
                    # case 2: tie
                    elif hand_score == opposite_hand_score:
                        money = money + 3
                        msg = "Tie"
                        update_state = 2
                    # case 3: lose
                    else:
                        msg = "Lose"
                        update_state = 2
                # if cards can be flipped more,
                else:
                    update_state = 1
                        
            elif on_die_button(click_pos) or auto_die:
                msg = "Fold, go on to the next"
                update_state = 3

    # show cards and info message including sound
    update_cards_and_sounds()

    pygame.time.delay(200)