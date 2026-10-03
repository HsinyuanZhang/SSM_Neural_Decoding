`timescale 1ns/1ps
module tb_real2x2_ssm_engine;
  parameter integer STATE_W = 16;
  logic clk=0, rst_n=0, valid; logic ready;
  logic signed [15:0] a00,a01,a10,a11,b0,b1;
  logic signed [STATE_W-1:0] u; logic signed [STATE_W-1:0] p,q;
  logic [31:0] saturation_count;
  integer fd, rc, line, eu, ep, eq, es; integer failures=0;
  real2x2_ssm_engine #(.STATE_W(STATE_W)) dut (.*);
  always #5 clk=~clk;
  initial begin
    valid=0; a00=15565; a01=-2344; a10=2344; a11=15565; b0=0; b1=16384; u=0;
    repeat(2) @(posedge clk); rst_n=1;
    fd=$fopen("hardware/fixedpoint_vectors.txt", "r");
    if(fd==0) begin $display("cannot open vector file"); $finish_and_return(2); end
    line=0;
    while (!$feof(fd)) begin
      rc=$fscanf(fd, "%d %d %d %d\n", eu,ep,eq,es);
      if(rc==4) begin
        u=eu; valid=1; @(posedge clk); #1;
        if(p!==ep || q!==eq || saturation_count!==es) begin
          $display("FAIL line=%0d u=%0d got p/q/sat=%0d/%0d/%0d expected=%0d/%0d/%0d",line,eu,p,q,saturation_count,ep,eq,es);
          failures=failures+1;
        end
        line=line+1;
      end
    end
    valid=0; $fclose(fd);
    if(failures==0) $display("PASS %0d fixed-point vectors",line);
    else $display("FAIL %0d mismatches",failures);
    $finish;
  end
endmodule
