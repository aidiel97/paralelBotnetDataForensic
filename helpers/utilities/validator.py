def validateInput(input):
  try:
    input = int(input)-1
    isValid = True
  except ValueError:
    isValid = False
  return isValid, input