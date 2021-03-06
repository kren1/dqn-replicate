#!/usr/bin/python3
import numpy as np
from random import choice
import distance
#Representation charcahter -> index of one hot vector
# digits 1-9 -> 0-8
# + -> 9 
# - -> 10
# ( -> 11
# ) -> 12
digitMap = [
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"+",
"-",
"(",
")"
]

digitLookUp = { digit: index for index, digit in enumerate(digitMap)}

def find(vector):
  return np.nonzero(vector)[0][0]

class NNExpr:
  def __init__(self, expression, window_size=8):
    expression = expression.replace(" ","")
    self.size = len(expression) 
    self.window_left = 0
    self.window_right = window_size
    self.cursor_pos = 0
    self.arr = np.zeros((self.size, 14))
    self.arr[self.cursor_pos, 0] = 1
    for ind, c in enumerate(expression):
      self.arr[ind, digitLookUp[c] + 1] = 1
  def cursorLeft(self):
    cursor_pos = find(self.arr[:,0])
    self.arr[cursor_pos,0] = 0
    cursor_pos -= 1
    cursor_pos = self.size - 1 if cursor_pos == 0 else cursor_pos
    self.arr[cursor_pos,0] = 1
  def cursorRight(self):
    cursor_pos = find(self.arr[:,0])
    self.arr[cursor_pos,0] = 0
    cursor_pos += 1
    cursor_pos = 0 if cursor_pos == self.size else cursor_pos
    self.arr[cursor_pos,0] = 1
  def replace(self):
    cursor_pos = find(self.arr[:,0])
    curr_digit = find(self.arr[cursor_pos, 1:])
    self.arr[cursor_pos, 1:] = 0
    curr_digit += 2
    curr_digit = 1 if curr_digit >= len(digitMap) else curr_digit
    self.arr[cursor_pos, curr_digit] = 1
  def panLeft(self):
   pan_diff = 0 if self.window_left  == 0 else 1 
   self.window_right -= pan_diff
   self.window_left -= pan_diff
  def panRight(self):
   pan_diff = 0 if self.window_right  == self.size else 1 
   self.window_right += pan_diff
   self.window_left += pan_diff
  def insert(self):
    cursor_pos = find(self.arr[:,0])
    new_arr = np.zeros((self.size + 6, 14))
    new_arr[:cursor_pos, :] = self.arr[:cursor_pos, :]
    new_arr[cursor_pos, 12] = 1
    new_arr[cursor_pos + 1, 1] = 1
    new_arr[cursor_pos + 2, 10] = 1
    new_arr[cursor_pos + 3, 1] = 1
    new_arr[cursor_pos + 4, 13] = 1
    new_arr[cursor_pos + 5, 10] = 1
    new_arr[cursor_pos + 6:, :] = self.arr[cursor_pos:, :]
    self.arr = new_arr
    self.size += 6
  def __str__(self):
    return "".join(list(map(lambda a: digitMap[find(a[1:])], self.arr)))
  def print(self):
    line1 = ""  
    line2 = ""  
    for vector in self.arr[self.window_left:self.window_right]:
      line1 += digitMap[find(vector[1:])] + " "
      line2 += "^ " if vector[0] == 1 else "  " 
    return line1 + '\n' + line2
  def evalExpr(self):
    return eval(str(self))

def generateExpressions(seed="1+1"):
  while True:
    e = NNExpr(seed)
    h = Harness(e)
    i = 0
    while h.act(choice([0,0,0,0,0,2,2,2,2,5]), False) < 1:
      i += 1
      if i > 20:
          i = 0
          h = Harness(NNExpr(seed))
    seed = str(h.nnexpr)
    h = Harness(NNExpr(seed))
    yield seed
    



class Harness:
  def __init__(self, nnexpr):
    self.nnexpr = nnexpr
    self.actions = {
       0: nnexpr.cursorLeft,
       1: nnexpr.cursorRight,
       2: nnexpr.replace,
       3: nnexpr.panLeft,
       4: nnexpr.panRight,
       5: nnexpr.insert
     }
    self.initial_expr = str(nnexpr)
    self.initial_value = nnexpr.evalExpr()
  def act(self, action, equal=True):
    self.actions[action]()
    try:
      value = self.nnexpr.evalExpr()
    except SyntaxError:
      return -1 #Failed to compile
    except TypeError:
      return -1 # eg. expression (1+1)(1+)
    if equal and value != self.initial_value:
      return -1 #Wrong value
    return distance.levenshtein(self.initial_expr, str(self.nnexpr))
      




import curses
def main(stdscr):
  e = NNExpr("1+1+1-2+1")
  stdscr.addstr(0,0,"Press q to quit")
  stdscr.addstr(1,0,e.print())
  while True:
    c = stdscr.getch()
    if c == ord('q'):
      break
    elif c == curses.KEY_LEFT:
      e.cursorLeft()
      stdscr.addstr(1,0,e.print())
    elif c == curses.KEY_RIGHT:
      e.cursorRight()
      stdscr.addstr(1,0,e.print())
    elif c == ord('e'):
      stdscr.addstr(4,0,str(e.evalExpr()))
    elif c == ord('i'):
      e.insert()
      stdscr.addstr(1,0,e.print())
    elif c == ord('l'):
      e.panRight()
      stdscr.addstr(1,0,e.print())
    elif c == ord('h'):
      e.panLeft()
      stdscr.addstr(1,0,e.print())
    elif c == ord('r'):
      e.replace()
      stdscr.addstr(1,0,e.print())


if __name__ == '__main__':
   curses.wrapper(main)
