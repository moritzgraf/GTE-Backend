#!/usr/bin/env wolframscript
Unprotect[Inequality];

                Format[HoldPattern @ Inequality[a__], InputForm] := Module[{res = HoldForm[a], rel},
                    rel = List @@ Replace[res[[2 ;; -1 ;; 2]],
                        {
                        Less -> " < ",
                        LessEqual -> " <= ",
                        Greater -> " > ",
                        GreaterEqual -> " >= ",
                        Equal -> " == ",
                        Unequal -> " != "
                        },
                        {1}
                    ];
                    res = InputForm /@ res;
                    res[[2 ;; -1 ;; 2]] = rel;
                    Replace[res, HoldForm[z__] :> OutputForm @ HoldForm @ SequenceForm[z]]
                    ]

                Protect[Inequality];

Print["Analysis of equilibria in /home/mg_linux/Master/GTE_SE/GTE-Backend/sequential_solver/solver/example_input/game_to_solve.ef
 filtered using full nash rationality"]
SE = {}
NE = {}
setime = 0
{nashtime, nashresult} = AbsoluteTiming[CylindricalDecomposition[I1A1p >= 0 && (1-I1A1p) >= 0 && I2A1p >= 0 && (1-I2A1p) >= 0 && I3A1p >= 0 && (1-I3A1p) >= 0 && I1A1p+(1-I1A1p)== 1 && I2A1p+(1-I2A1p)== 1 && I3A1p+(1-I3A1p)== 1 && (1*(1-1)*(1-I2A1p)+3*1*I3A1p+0*1*(1-I3A1p)+4*(1-1)*I2A1p*I3A1p+0*(1-1)*I2A1p*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p)) <= 0 && I1A1p*((1*(1-1)*(1-I2A1p)+3*1*I3A1p+0*1*(1-I3A1p)+4*(1-1)*I2A1p*I3A1p+0*(1-1)*I2A1p*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p))) == 0 && (1*1*(1-I2A1p)+3*0*I3A1p+0*0*(1-I3A1p)+4*1*I2A1p*I3A1p+0*1*I2A1p*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p)) <= 0 && (1-I1A1p)*((1*1*(1-I2A1p)+3*0*I3A1p+0*0*(1-I3A1p)+4*1*I2A1p*I3A1p+0*1*I2A1p*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p))) == 0 && (1*(1-I1A1p)*(1-1)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*1*I3A1p+0*(1-I1A1p)*1*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p)) <= 0 && I2A1p*((1*(1-I1A1p)*(1-1)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*1*I3A1p+0*(1-I1A1p)*1*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p))) == 0 && (1*(1-I1A1p)*1+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*0*I3A1p+0*(1-I1A1p)*0*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p)) <= 0 && (1-I2A1p)*((1*(1-I1A1p)*1+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*0*I3A1p+0*(1-I1A1p)*0*(1-I3A1p))-(1*(1-I1A1p)*(1-I2A1p)+3*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+4*(1-I1A1p)*I2A1p*I3A1p+0*(1-I1A1p)*I2A1p*(1-I3A1p))) == 0 && (1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*1+0*I1A1p*(1-1)+0*(1-I1A1p)*I2A1p*1+1*(1-I1A1p)*I2A1p*(1-1))-(1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+0*(1-I1A1p)*I2A1p*I3A1p+1*(1-I1A1p)*I2A1p*(1-I3A1p)) <= 0 && I3A1p*((1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*1+0*I1A1p*(1-1)+0*(1-I1A1p)*I2A1p*1+1*(1-I1A1p)*I2A1p*(1-1))-(1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+0*(1-I1A1p)*I2A1p*I3A1p+1*(1-I1A1p)*I2A1p*(1-I3A1p))) == 0 && (1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*0+0*I1A1p*1+0*(1-I1A1p)*I2A1p*0+1*(1-I1A1p)*I2A1p*1)-(1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+0*(1-I1A1p)*I2A1p*I3A1p+1*(1-I1A1p)*I2A1p*(1-I3A1p)) <= 0 && (1-I3A1p)*((1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*0+0*I1A1p*1+0*(1-I1A1p)*I2A1p*0+1*(1-I1A1p)*I2A1p*1)-(1*(1-I1A1p)*(1-I2A1p)+2*I1A1p*I3A1p+0*I1A1p*(1-I3A1p)+0*(1-I1A1p)*I2A1p*I3A1p+1*(1-I1A1p)*I2A1p*(1-I3A1p))) == 0, {I1A1p, I2A1p, I3A1p}, "Function"]]
nes = BooleanConvert[Normal[nashresult]];
NE = If[Head[nes] == Or, List @@ nes, {nes}, {nes}];
{time, result} = AbsoluteTiming[CylindricalDecomposition[ nashresult && (1) >= 0 && (1) >= 0 && I3N1b >= 0 && (1-I3N1b) >= 0 && (1)== 1 && (1)== 1 && I3N1b+(1-I3N1b)== 1 && ((1)*(3*I3A1p+0*(1-I3A1p)))-((1)*I1A1p*(3*I3A1p+0*(1-I3A1p))+(1)*(1-I1A1p)*(1*(1-I2A1p)+4*I2A1p*I3A1p+0*I2A1p*(1-I3A1p)))<= 0 && I1A1p(((1)*(3*I3A1p+0*(1-I3A1p)))-((1)*I1A1p*(3*I3A1p+0*(1-I3A1p))+(1)*(1-I1A1p)*(1*(1-I2A1p)+4*I2A1p*I3A1p+0*I2A1p*(1-I3A1p)))) == 0 && ((1)*(1*(1-I2A1p)+4*I2A1p*I3A1p+0*I2A1p*(1-I3A1p)))-((1)*I1A1p*(3*I3A1p+0*(1-I3A1p))+(1)*(1-I1A1p)*(1*(1-I2A1p)+4*I2A1p*I3A1p+0*I2A1p*(1-I3A1p)))<= 0 && (1-I1A1p)(((1)*(1*(1-I2A1p)+4*I2A1p*I3A1p+0*I2A1p*(1-I3A1p)))-((1)*I1A1p*(3*I3A1p+0*(1-I3A1p))+(1)*(1-I1A1p)*(1*(1-I2A1p)+4*I2A1p*I3A1p+0*I2A1p*(1-I3A1p)))) == 0 && ((1)*(4*I3A1p+0*(1-I3A1p)))-((1)*I2A1p*(4*I3A1p+0*(1-I3A1p))+(1)*(1-I2A1p)*(1))<= 0 && I2A1p(((1)*(4*I3A1p+0*(1-I3A1p)))-((1)*I2A1p*(4*I3A1p+0*(1-I3A1p))+(1)*(1-I2A1p)*(1))) == 0 && ((1)*(1))-((1)*I2A1p*(4*I3A1p+0*(1-I3A1p))+(1)*(1-I2A1p)*(1))<= 0 && (1-I2A1p)(((1)*(1))-((1)*I2A1p*(4*I3A1p+0*(1-I3A1p))+(1)*(1-I2A1p)*(1))) == 0 && (I3N1b*(2)+(1-I3N1b)*(0))-(I3N1b*I3A1p*(2)+(1-I3N1b)*I3A1p*(0)+I3N1b*(1-I3A1p)*(0)+(1-I3N1b)*(1-I3A1p)*(1))<= 0 && I3A1p((I3N1b*(2)+(1-I3N1b)*(0))-(I3N1b*I3A1p*(2)+(1-I3N1b)*I3A1p*(0)+I3N1b*(1-I3A1p)*(0)+(1-I3N1b)*(1-I3A1p)*(1))) == 0 && (I3N1b*(0)+(1-I3N1b)*(1))-(I3N1b*I3A1p*(2)+(1-I3N1b)*I3A1p*(0)+I3N1b*(1-I3A1p)*(0)+(1-I3N1b)*(1-I3A1p)*(1))<= 0 && (1-I3A1p)((I3N1b*(0)+(1-I3N1b)*(1))-(I3N1b*I3A1p*(2)+(1-I3N1b)*I3A1p*(0)+I3N1b*(1-I3A1p)*(0)+(1-I3N1b)*(1-I3A1p)*(1))) == 0 && I3N1b^1 * 1^1 * (1-I1A1p)^1 * I2A1p^1 == (1-I3N1b)^1 * I1A1p^1 * 1^1 * 1^1, {I1A1p, I2A1p, I3A1p, I3N1b}]];
 newEQ = BooleanConvert[Simplify[result]];
SE = Join[SE, If[Head[newEQ] == Or, List @@ newEQ, {newEQ}, {newEQ}]];
Print["Equations generated in ", 0.004194974899291992, "s (python)"]
Print["

Nash Equilibria, calculated in ", nashtime, "s (mathematica):"]
For[i=1,i<=Length[NE],i++, Print[Format[NE[[i]], InputForm]]]
 
Print["

Sequential Equilibria, calculated in ", time, "s (mathematica): "]

For[i=1,i<=Length[SE],i++, Print[Format[CylindricalDecomposition[SE[[i]], {I1A1p, I2A1p, I3A1p, I3N1b}], InputForm]]]
Print["
Total required time: ", (0.004194974899291992+ nashtime + time), "s"]