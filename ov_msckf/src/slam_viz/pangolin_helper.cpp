
#include "pangolin_helper.h"

#include <fstream>

namespace slam_viz {

namespace pangolin_helper {

std::shared_ptr<pangolin::GlFont> chinese_font;

namespace internal {

bool checkFileExistence(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}

bool loadChineseFont(const std::string& font_file) {
  std::cout << "loadChineseFont(): "
            << "Loading Chinese font from file '" << font_file << "'. "
            << "The process might be slow, please wait ..." << std::endl;
  chinese_font = std::make_shared<pangolin::GlFont>(font_file, 18.0, 5120, 5120);
  if (chinese_font) {
    return true;
  } else {
    std::cerr << "loadChineseFont(): Failed to load font from file '"
              << font_file << "'!" << std::endl;
    return false;
  }
}

void drawTextLine(const TextLine& line) {
  glColor4ub(line.color[0], line.color[1], line.color[2], line.color[3]);
  line.font->Text(line.text).Draw();
}

void drawMultiTextLines(const TextLines& lines) {
  for (const TextLine& line : lines) {
    glColor4ub(line.color[0], line.color[1], line.color[2], line.color[3]);
    line.font->Text(line.text).Draw();
    glTranslatef(0, -20.0 /*assuming the default font height is 18.0*/, 0);
  }
}

void drawText(std::function<void()> do_draw,
              const Eigen::Vector3f& translation,
              const Eigen::Matrix3f& mult_rotation,
              float scale) {
  glPushMatrix();
  Eigen::Matrix4f mult_matrix = Eigen::Matrix4f::Identity();
  if (scale != 1.0) {
    mult_matrix.block(0,0,3,3) = scale * mult_rotation;
  } else {
    mult_matrix.block(0,0,3,3) = mult_rotation;
  }
  mult_matrix.block(0,3,3,1) = translation;
  glMultMatrixf(mult_matrix.data());

  if (do_draw) {
    do_draw();
  }

  glPopMatrix();  
}


void drawTextFacingScreen(
    std::function<void()> do_draw,
    const Eigen::Vector3f& translation,
    float scale) {

  GLint    view[4];
  glGetIntegerv(GL_VIEWPORT, view);
  // std::cout << "drawTextFacingScreen(): Debug GL_VIEWPORT: " 
  //           << view[0] << ", " << view[1] << ", " 
  //           << view[2] << ", " << view[3] << std::endl;
  // float cur_view_width = view[2];
  // float cur_view_height = view[3];

  GLdouble scrn[3];
  // find object point (x,y,z)' in pixel coords (of WINDOW, not of VIEW)
  GLdouble projection[16];
  GLdouble modelview[16];
  
#ifdef HAVE_GLES_2
  std::copy(glEngine().projection.top().m, glEngine().projection.top().m+16, projection);
  std::copy(glEngine().modelview.top().m, glEngine().modelview.top().m+16, modelview);
#else
  glGetDoublev(GL_PROJECTION_MATRIX, projection);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
#endif

  // Compute the pixel coordinates of the top-left corner to display text
  pangolin::glProject(
      translation.x(), translation.y(), translation.z(),
      modelview, projection, view,
      scrn, scrn + 1, scrn + 2);

  // std::cout << "drawTextFacingScreen(): Debug scrn: " 
  //           << scrn[0] << ", " << scrn[1] << ", " 
  //           << scrn[2] << std::endl;

  // Save current state
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  // glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // See the implementation in `pangolin::SetWindowOrthographic()`;
  {
    // We'll set an arbitrary viewport with known dimensions
    // >= window dimensions so we can draw in pixel units.

    GLint dims[2];
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS,dims);
    glViewport(0,0,dims[0], dims[1]);

    // std::cout << "drawTextFacingScreen(): Debug GL_MAX_VIEWPORT_DIMS: " 
    //           << dims[0] << ", " << dims[1] << std::endl;
    // // dims[0] = dims[1] = 16384

    glMatrixMode(GL_PROJECTION);
    pangolin::ProjectionMatrixOrthographic(-0.5, dims[0]-0.5, -0.5, dims[1]-0.5, -1.0, 1.0).Load();
    glMatrixMode(GL_MODELVIEW);
  }

  if (scale != 1.0) {
    Eigen::Matrix4f mult_matrix = Eigen::Matrix4f::Identity();
    mult_matrix.block(0,0,3,3) = scale * Eigen::Matrix3f::Identity();
    mult_matrix.block(0,3,3,1) = Eigen::Vector3f(scrn[0], scrn[1], scrn[2]);
    glMultMatrixf(mult_matrix.data());
  } else {
    // glTranslatef(scrn[0], scrn[1], scrn[2]);

    // Using std::floor ensures the text are drawing from interger pixel coordinates,
    // which brings better display effect.
    glTranslatef(
        std::floor((GLfloat)scrn[0]), std::floor((GLfloat)scrn[1]), (GLfloat)scrn[2]);
  }

  if (do_draw) {
    do_draw();
  }

  // Restore viewport & matrices
  glViewport(view[0],view[1],view[2],view[3]);
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void drawTextInViewCoord(
    std::function<void()> do_draw,
    int start_pix_x,  int start_pix_y,
    float scale) {

  GLint    view[4];
  glGetIntegerv(GL_VIEWPORT, view);
  // std::cout << "drawTextInViewCoord(): Debug GL_VIEWPORT: " 
  //           << view[0] << ", " << view[1] << ", " 
  //           << view[2] << ", " << view[3] << std::endl;
  float cur_view_width = view[2];  // current_view.v.w
  float cur_view_height = view[3];  // current_view.v.h

  // deal with negative input for convenience
  if (start_pix_x < 0) {
    start_pix_x = cur_view_width + start_pix_x;
  }
  if (start_pix_y < 0) {
    start_pix_y = cur_view_height + start_pix_y;
  }

  // Save current state
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  // glLoadIdentity();
  pangolin::ProjectionMatrixOrthographic(
      0, cur_view_width,
      0, cur_view_height,
      -1.0, 1.0).Load();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  float x = start_pix_x;
  float y = start_pix_y;
  float z = 1.0;  // z can be orbitrary

  if (scale != 1.0) {
    Eigen::Matrix4f mult_matrix = Eigen::Matrix4f::Identity();
    mult_matrix.block(0,0,3,3) = scale * Eigen::Matrix3f::Identity();
    mult_matrix.block(0,3,3,1) = Eigen::Vector3f(x, y, z);
    glMultMatrixf(mult_matrix.data());
  } else {
    glTranslatef(x, y, z);
  }

  if (do_draw) {
    do_draw();
  }

  // Restore viewport & matrices
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}


void drawTextInWindowCoord(
    std::function<void()> do_draw,
    int start_pix_x,  int start_pix_y,
    float scale) {

  // See GlText::DrawWindow()
  
  // Backup viewport & matrices
  GLint    view[4];
  glGetIntegerv(GL_VIEWPORT, view );
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  // glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // // deal with negative input for convenience
  // // todo: how to get window's size?
  // if (start_pix_x < 0) {
  //   start_pix_x = cur_window_width + start_pix_x;
  // }
  // if (start_pix_y < 0) {
  //   start_pix_y = cur_window_height + start_pix_y;
  // }

  // See the implementation in `pangolin::SetWindowOrthographic()`;
  {
    // We'll set an arbitrary viewport with known dimensions
    // >= window dimensions so we can draw in pixel units.
    GLint dims[2];
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS,dims);
    glViewport(0,0,dims[0], dims[1]);

    glMatrixMode(GL_PROJECTION);
    pangolin::ProjectionMatrixOrthographic(-0.5, dims[0]-0.5, -0.5, dims[1]-0.5, -1.0, 1.0).Load();
    glMatrixMode(GL_MODELVIEW);
  }

  float x = start_pix_x;
  float y = start_pix_y;
  float z = 1.0;  // z can be orbitrary

  if (scale != 1.0) {
    Eigen::Matrix4f mult_matrix = Eigen::Matrix4f::Identity();
    mult_matrix.block(0,0,3,3) = scale * Eigen::Matrix3f::Identity();
    mult_matrix.block(0,3,3,1) = Eigen::Vector3f(x, y, z);
    glMultMatrixf(mult_matrix.data());
  } else {
    glTranslatef(x, y, z);
  }

  if (do_draw) {
    do_draw();
  }

  // Restore viewport & matrices
  glViewport(view[0],view[1],view[2],view[3]);
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

}  // namespace internal

bool loadChineseFont() {
  static bool already_invoked = false;
  if (already_invoked) {
    std::cout << "loadChineseFont(): Repeated calls to loadChineseFont()"
              << std::endl;
    if (chinese_font) {
      return true;
    } else {
      return false;
    }
  }

  already_invoked = true;
  std::string font_file;

  // Try the following path in order:
  //   1. ${PANGOLIN_CHINESE_FONTFILE}
  //   2. "SmileySans-Oblique.ttf"
  //   3. "/usr/share/fonts/truetype/smiley-sans/SmileySans-Oblique.ttf"
  //   4. "/usr/local/share/fonts/truetype/smiley-sans/SmileySans-Oblique.ttf"
  //   5. "${this_dir}/pangolin_fonts/Chinese_deyihei/SmileySans-Oblique.ttf"

  const char* env_value = std::getenv("PANGOLIN_CHINESE_FONTFILE");
  if (env_value) {
    font_file = env_value;
    if (!font_file.empty()) {
      if (internal::checkFileExistence(font_file)) {
        return internal::loadChineseFont(font_file);
      }
    }
  }

  font_file = "SmileySans-Oblique.ttf";
  if (internal::checkFileExistence(font_file)) {
    return internal::loadChineseFont(font_file);
  }

  font_file = "/usr/share/fonts/truetype/smiley-sans/SmileySans-Oblique.ttf";
  if (internal::checkFileExistence(font_file)) {
    return internal::loadChineseFont(font_file);
  }

  font_file = "/usr/local/share/fonts/truetype/smiley-sans/SmileySans-Oblique.ttf";  // NOLINT
  if (internal::checkFileExistence(font_file)) {
    return internal::loadChineseFont(font_file);
  }

  std::string this_file = __FILE__;
  std::string this_dir = this_file.substr(0, this_file.rfind("/"));
  font_file = this_dir + "/pangolin_fonts/Chinese_deyihei/SmileySans-Oblique.ttf";  // NOLINT
  if (internal::checkFileExistence(font_file)) {
    return internal::loadChineseFont(font_file);
  }

  std::cerr << "loadChineseFont(): "
            << "Can't find the font file 'SmileySans-Oblique.ttf'! "
            << "Please install the font by running "
            << "'sudo apt-get install fonts-smiley-sans'  OR "
            << "'wget http://ports.ubuntu.com/pool/universe/f/fonts-smiley-sans/fonts-smiley-sans_1.1.1-1_all.deb "  // NOLINT
            << "&& sudo dpkg -i fonts-smiley-sans_1.1.1-1_all.deb'."
            << std::endl;
  return false;
}

pangolin::GlFont* getChineseFont() {
  if (chinese_font) {
    return chinese_font.get();
  } else {
    return &pangolin::default_font();
  }
}

TextLine::TextLine(const std::string& str, bool hightlight, pangolin::GlFont* ifont)
    : text(str), font(ifont) {
  if (hightlight) {
    color[0] = 255;
    color[1] = 0;
    color[2] = 0;
    color[3] = 255;
  } else {
    color[0] = 127;
    color[1] = 127;
    color[2] = 127;
    color[3] = 255;
  }
}

TextLine::TextLine(
    const std::string& str,
    uint8_t r, uint8_t g, uint8_t b, uint8_t a,
    pangolin::GlFont* ifont)
    : text(str), font(ifont) {
  color[0] = r;
  color[1] = g;
  color[2] = b;
  color[3] = a;
}

void drawMultiTextLines(
    const TextLines& lines,
    const Eigen::Vector3f& translation,
    const Eigen::Matrix3f& mult_rotation,
    float scale) {
  internal::drawText([&](){
    internal::drawMultiTextLines(lines);
  }, translation, mult_rotation, scale);
}

void drawMultiTextLinesFacingScreen(
    const TextLines& lines,
    const Eigen::Vector3f& translation,
    float scale) {

  internal::drawTextFacingScreen([&](){
    internal::drawMultiTextLines(lines);
  }, translation, scale);
}

void drawMultiTextLinesInViewCoord(
    const TextLines& lines,
    int start_pix_x,  int start_pix_y,
    float scale) {

  internal::drawTextInViewCoord([&](){
    internal::drawMultiTextLines(lines);
  }, start_pix_x, start_pix_y, scale);
}

void drawMultiTextLinesInWindowCoord(
    const TextLines& lines,
    int start_pix_x,  int start_pix_y,
    float scale) {

  internal::drawTextInWindowCoord([&](){
    internal::drawMultiTextLines(lines);
  }, start_pix_x, start_pix_y, scale);
}



void drawTextLine(
    const TextLine& line,
    const Eigen::Vector3f& translation,
    const Eigen::Matrix3f& mult_rotation,
    float scale) {
  internal::drawText([&](){
    internal::drawTextLine(line);
  }, translation, mult_rotation, scale);
}

void drawTextLineFacingScreen(
    const TextLine& line,
    const Eigen::Vector3f& translation,
    float scale) {

  internal::drawTextFacingScreen([&](){
    internal::drawTextLine(line);
  }, translation, scale);
}

void drawTextLineInViewCoord(
    const TextLine& line,
    int start_pix_x,  int start_pix_y,
    float scale) {
  internal::drawTextInViewCoord([&](){
    internal::drawTextLine(line);
  }, start_pix_x, start_pix_y, scale);
}

void drawTextLineInWindowCoord(
    const TextLine& line,
    int start_pix_x,  int start_pix_y,
    float scale) {
  internal::drawTextInWindowCoord([&](){
    internal::drawTextLine(line);
  }, start_pix_x, start_pix_y, scale);
}

void drawTrianglePose() {
  
}

} // namespace pangolin_helper

} // namespace slam_viz

