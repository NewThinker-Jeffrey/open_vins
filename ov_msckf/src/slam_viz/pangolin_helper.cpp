
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
  glColor4ub(line.c.r, line.c.g, line.c.b, line.c.a);
  line.font->Text(line.text).Draw();
}

void drawMultiTextLines(const TextLines& lines) {
  for (const TextLine& line : lines) {
    glColor4ub(line.c.r, line.c.g, line.c.b, line.c.a);
    line.font->Text(line.text).Draw();
    glTranslatef(0, -20.0 /*assuming the default font height is 18.0*/, 0);
  }
}

void drawText(std::function<void()> do_draw,
              const Eigen::Vector3f& translation,
              const Eigen::Matrix3f& mult_rotation,
              float scale) {

  multMatrixfAndDraw(
      makeMatrixf(translation, mult_rotation, scale), do_draw);

  // glPushMatrix();
  // Eigen::Matrix4f mult_matrix = Eigen::Matrix4f::Identity();
  // if (scale != 1.0) {
  //   mult_matrix.block(0,0,3,3) = scale * mult_rotation;
  // } else {
  //   mult_matrix.block(0,0,3,3) = mult_rotation;
  // }
  // mult_matrix.block(0,3,3,1) = translation;
  // glMultMatrixf(mult_matrix.data());

  // if (do_draw) {
  //   do_draw();
  // }

  // glPopMatrix();  
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
    c = Color(255, 0, 0, 255);
  } else {
    c = Color(127, 127, 127, 255);
  }
}

TextLine::TextLine(
    const std::string& str,
    Color _c,
    pangolin::GlFont* ifont)
    : text(str), c(_c), font(ifont) {}

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


void drawGrids2D(
    int center_x, int center_y, int n_x, int n_y,
    Color c, float line_width) {

  int x_begin = center_x - n_x / 2;
  int y_begin = center_y - n_y / 2;

  glLineWidth(line_width);
  glColor4ub(c.r, c.g, c.b, c.a);

  glBegin(GL_LINES);
  for (int i=0; i<n_x; i++) {
    // draw the ith vertical line
    glVertex3f(x_begin + i, y_begin, 0);
    glVertex3f(x_begin + i, y_begin + n_y, 0);
  }

  for (int i=0; i<n_y; i++) {
    // draw the ith horizontal line
    glVertex3f(x_begin, y_begin + i, 0);
    glVertex3f(x_begin + n_x, y_begin + i, 0);
  }

  glEnd();
}

void drawFrame(float axis_len, float line_width, uint8_t alpha) {

  glLineWidth(line_width);

  glBegin(GL_LINES);

  glColor4ub(255, 0, 0, alpha);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(axis_len, 0.0f, 0.0f);

  glColor4ub(0, 255, 0, alpha);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, axis_len, 0.0f);

  glColor4ub(0, 0, 255, alpha);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, axis_len);

  glEnd();
}

void drawCamera(double s, Color c, float line_width) {

  float half_w = 1.0f * s;
  float half_h = 0.7f * s;
  float z = 1.0f * s;

  glLineWidth(line_width);
  glColor4ub(c.r, c.g, c.b, c.a);

  // glBegin(GL_LINE_STRIP);
  glBegin(GL_LINE_LOOP);
  glVertex3f(-half_w, -half_h, z);
  glVertex3f( half_w, -half_h, z);
  glVertex3f( half_w,  half_h, z);
  glVertex3f(-half_w,  half_h, z);
  glEnd();

  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(-half_w, -half_h, z);
  glVertex3f(0, 0, 0);
  glVertex3f( half_w, -half_h, z);
  glVertex3f(0, 0, 0);
  glVertex3f( half_w,  half_h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-half_w,  half_h, z);
  glEnd();
}

void drawVehicle(
    double s,
    const Eigen::Vector3f& front,
    const Eigen::Vector3f& up,
    Color c, float line_width) {

  glLineWidth(line_width);
  glColor4ub(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_LOOP);
  double half_w = 0.4 * s;
  double l = 1.0 * s;
  Eigen::Vector3f left = up.cross(front);
  Eigen::Vector3f pl =   half_w * left - l * front;
  Eigen::Vector3f pr = - half_w * left - l * front;
  glVertex3f(pl.x(), pl.y(), pl.z());
  glVertex3f(pr.x(), pr.y(), pr.z());
  glVertex3f(      0, 0,  0);
  glEnd();
}

void drawCvImageOnView(
    const cv::Mat& img_in,
    pangolin::View& view,
    bool need_bgr2rgb) {
  cv::Mat img;
  if (img_in.channels() == 1) {
    cv::cvtColor(img_in, img, cv::COLOR_GRAY2RGB);
  } else if (img_in.channels() == 3) {
    if (need_bgr2rgb) {
      cv::cvtColor(img_in, img, cv::COLOR_BGR2RGB);
    } else {
      img = img_in.clone();
    }
  }

  // cv::flip(img, img, 0);  // this works with RenderToViewport()

  pangolin::GlTexture imageTexture(img.cols, img.rows, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
  imageTexture.Upload(img.ptr<uchar>(), GL_RGB, GL_UNSIGNED_BYTE);

  view.SetAspect(img.cols/(float)img.rows);
  view.Activate();
  glColor4ub(255, 255, 255, 255);

  // imageTexture.RenderToViewport();  // This needs cv::flip(img, img, 0) before hand
  imageTexture.RenderToViewportFlipY();
  // imageTexture.RenderToViewportFlipXFlipY();
}


Eigen::Matrix4f makeMatrixf(
    const Eigen::Vector3f& translation,
    const Eigen::Matrix3f& rotation,
    float scale) {
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  if (scale != 1.0) {
    matrix.block(0,0,3,3) = scale * rotation;
  } else {
    matrix.block(0,0,3,3) = rotation;
  }
  matrix.block(0,3,3,1) = translation;
  return matrix;
}

void multMatrixfAndDraw(
    const Eigen::Matrix4f& matrix,
    std::function<void()> do_draw) {
  
  glPushMatrix();
  glMultMatrixf(matrix.data());

  if (do_draw) {
    do_draw();
  }
  glPopMatrix();
}

pangolin::OpenGlMatrix
makeGlMatrix(const Eigen::Matrix4f& eigen_float_mat) {
  pangolin::OpenGlMatrix ret;
  for (int i = 0; i<4; i++) {
    ret.m[4*i] = eigen_float_mat(0,i);
    ret.m[4*i+1] = eigen_float_mat(1,i);
    ret.m[4*i+2] = eigen_float_mat(2,i);
    ret.m[4*i+3] = eigen_float_mat(3,i);
  }
  return ret;
}

} // namespace pangolin_helper

} // namespace slam_viz

