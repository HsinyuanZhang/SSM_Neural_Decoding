// Signed fixed-point real 2x2 SSM engine.  Coefficients are Q1.14 int16.
// Arithmetic contract: sum all products wide, round nearest/ties-away from zero
// once by 14 bits, then saturate independently into signed STATE_W state regs.
module real2x2_ssm_engine #(
  parameter integer STATE_W = 16
) (
  input  logic clk, input logic rst_n,
  input logic valid, output logic ready,
  input logic signed [15:0] a00, a01, a10, a11,
  input logic signed [15:0] b0, b1,
  input logic signed [STATE_W-1:0] u,
  output logic signed [STATE_W-1:0] p, q,
  output logic [31:0] saturation_count
);
  localparam logic signed [63:0] STATE_MAX = (64'sd1 <<< (STATE_W-1)) - 1;
  localparam logic signed [63:0] STATE_MIN = -(64'sd1 <<< (STATE_W-1));
  logic signed [63:0] p_sum, q_sum, p_rounded, q_rounded;
  logic p_sat, q_sat;

  function automatic logic signed [63:0] round_q14(input logic signed [63:0] x);
    logic signed [63:0] magnitude;
    begin
      // Explicit sign/magnitude form avoids implementation-dependent signed shifts on negatives.
      if (x >= 0) round_q14 = (x + 64'sd8192) >>> 14;
      else begin
        magnitude = -x;
        round_q14 = -((magnitude + 64'sd8192) >>> 14);
      end
    end
  endfunction

  function automatic logic signed [STATE_W-1:0] saturate_state(input logic signed [63:0] x);
    begin
      if (x > STATE_MAX) begin
        saturate_state = {1'b0, {(STATE_W-1){1'b1}}};
      end else if (x < STATE_MIN) begin
        saturate_state = {1'b1, {(STATE_W-1){1'b0}}};
      end else begin
        saturate_state = x[STATE_W-1:0];
      end
    end
  endfunction

  always_comb begin
    p_sum = a00 * p + a01 * q + b0 * u;
    q_sum = a10 * p + a11 * q + b1 * u;
    p_rounded = round_q14(p_sum);
    q_rounded = round_q14(q_sum);
    p_sat = (p_rounded > STATE_MAX) || (p_rounded < STATE_MIN);
    q_sat = (q_rounded > STATE_MAX) || (q_rounded < STATE_MIN);
  end
  assign ready = 1'b1; // one accepted recurrence per cycle; a shared-MAC variant may deassert this.

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin p <= '0; q <= '0; saturation_count <= '0; end
    else if (valid && ready) begin
      p <= saturate_state(p_rounded);
      q <= saturate_state(q_rounded);
      saturation_count <= saturation_count + p_sat + q_sat;
    end
  end
endmodule
