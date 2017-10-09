from antlr4 import *
from arithmeticLexer import arithmeticLexer
from arithmeticParser import arithmeticParser

def parse(inpt):
  lexer = arithmeticLexer(InputStream(inpt))
  stream = CommonTokenStream(lexer)
  parser = arithmeticParser(stream)
  tree = parser.expression()
  return tree, arithmeticLexer(InputStream(inpt))

class Expr:
  def __init__(self, left, right):
   self.left = left
   self.right = right
class AddExpr(Expr):
  def __str__(self):
   return "({} + {})".format(str(self.left), str(self.right))
class SubExpr(Expr):
  def __str__(self):
   return "({} - {})".format(str(self.left), str(self.right))


onePlusTwo = AddExpr(1,2)

import numpy as np

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

def find(vector):
  return np.nonzero(vector)[0][0]

class NNExpr:
  def __init__(self, size, window_size=8):
    self.arr = np.zeros((size, 14))
    if size % 2 == 0:
      size += 1
    self.arr[0,0] = 1
    self.arr[:,2] = 1
    for i in range(1, size, 2):
      self.arr[i,2] = 0
      self.arr[i,10] = 1
    self.size = size
    self.window_left = 0
    self.window_right = window_size
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
    line1 = ""  
    line2 = ""  
    for vector in self.arr[self.window_left:self.window_right]:
      line1 += digitMap[find(vector[1:])] + " "
      line2 += "^ " if vector[0] == 1 else "  " 
    return line1 + '\n' + line2
  def evalExpr(self):
    return eval("".join(list(map(lambda a: digitMap[find(a[1:])], self.arr))))



import curses
def main(stdscr):
  e = NNExpr(15)
  stdscr.addstr(0,0,"Press q to quit")
  stdscr.addstr(1,0,str(e))
  while True:
    c = stdscr.getch()
    if c == ord('q'):
      break
    elif c == curses.KEY_LEFT:
      e.cursorLeft()
      stdscr.addstr(1,0,str(e))
    elif c == curses.KEY_RIGHT:
      e.cursorRight()
      stdscr.addstr(1,0,str(e))
    elif c == ord('e'):
      stdscr.addstr(4,0,str(e.evalExpr()))
    elif c == ord('i'):
      e.insert()
      stdscr.addstr(1,0,str(e))
    elif c == ord('l'):
      e.panRight()
      stdscr.addstr(1,0,str(e))
    elif c == ord('h'):
      e.panLeft()
      stdscr.addstr(1,0,str(e))
    elif c == ord('r'):
      e.replace()
      stdscr.addstr(1,0,str(e))

curses.wrapper(main)
