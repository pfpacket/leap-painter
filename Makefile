TARGET   = leap-painter
OBJS	 = src/painter.o
LIBS     = opencv libleap
CXXFLAGS += -O2 -Wall -Wextra -std=c++17 $(shell pkg-config --cflags $(LIBS))
LDFLAGS  += $(shell pkg-config --libs $(LIBS))

all: $(TARGET)
rebuild: clean all

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) pic.jpg save.jpg

clean:
	$(RM) $(TARGET) $(OBJS)
